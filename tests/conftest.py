"""Shared fixtures for the Level 1 primitive smoke tests.

Contains a minimal fake LM emulating an HF causal LM with `.model.layers` —
needed for ActivationCollector hooks, without loading a real model.
"""

import pytest
import torch
import torch.nn as nn


class FakeLayer(nn.Module):
    """Minimal layer: Linear → tuple(hidden_states, ...) to imitate an HF Block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        # HF transformer layers return a tuple, the first element is hidden_states
        return (self.proj(x),)


class FakeInner(nn.Module):
    """Emulates `.model` inside an HF AutoModelForCausalLM."""

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)[0]
        return x


class _FakeConfig:
    """Emulates HF `model.config` — needed for auto-detect in Pipeline.from_pretrained."""

    def __init__(self, hidden_size: int, num_hidden_layers: int, vocab_size: int = 100):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size


class FakeLM(nn.Module):
    """Minimal model with `.model.layers` + `.config` — compatible with ActivationCollector
    and `MetaAttentionPipeline.from_pretrained(model=...)` (we pass a ready-made model so we
    don't download HF and don't need transformers)."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 4, vocab_size: int = 100):
        super().__init__()
        self.model = FakeInner(hidden_dim, num_layers)
        self.config = _FakeConfig(hidden_dim, num_layers, vocab_size)
        # HF-style: the forward interface accepts input_ids + attention_mask
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(
        self,
        x_or_input_ids: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        """Polymorphic forward:
          - `fake_lm(tensor[B, seq, hidden])` — Level 1 style, already-embedded → returns a tensor
          - `fake_lm(input_ids=tensor[B, seq])` — HF-style, embed inside → returns a tensor
          - `fake_lm(input_ids=..., labels=...)` — training → returns an HF-style object with .loss
        """
        # HF-style call with keyword input_ids
        if input_ids is not None:
            x = self.embed(input_ids)
        elif x_or_input_ids is not None:
            if x_or_input_ids.dtype in (torch.long, torch.int):
                x = self.embed(x_or_input_ids)
            else:
                # Already embedded — this is Level 1 style. We don't pass labels in this mode.
                return self.model(x_or_input_ids)
        else:
            x = kwargs.get("inputs_embeds")
            if x is None:
                raise ValueError("Need input_ids or inputs_embeds")

        hidden = self.model(x)

        # HF-style hidden_states: (embeddings, layer_1, …, layer_N). The stack is sequential,
        # so we re-run it layer by layer — used by goal_pool="tokens" (ChronoAnchor).
        if kwargs.get("output_hidden_states"):
            states = [x]
            h = x
            for layer in self.model.layers:
                h = layer(h)
                if isinstance(h, tuple):
                    h = h[0]
                states.append(h)
            out = type("Out", (), {})()
            out.hidden_states = tuple(states)
            out.logits = self.lm_head(hidden)
            out.last_hidden_state = hidden
            return out

        # If labels are passed — return an HF-style object with .loss + .logits
        if labels is not None:
            logits = self.lm_head(hidden)
            # Standard shift-by-one LM CE with masking ignore_index=-100
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            out = type("Out", (), {})()
            out.loss = loss
            out.logits = logits
            return out

        return hidden

    def generate(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 5,
        **kwargs,
    ) -> torch.Tensor:
        """Minimal autoregressive generation: greedy + lm_head on the final hidden."""
        cur = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                h = self.embed(cur)
                h = self.model(h)
                logits = self.lm_head(h[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                cur = torch.cat([cur, next_token], dim=1)
        return cur


class FakeTokenizer:
    """Minimal tokenizer: returns .input_ids, .attention_mask, decode yields a string.

    Supports batch (list[str]) with padding=True, max_length, truncation — for
    training collate.
    """

    def __init__(self, vocab_size: int = 100, pad_token_id: int = 0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token = "</s>"
        self.padding_side = "right"  # HF default; the collector switches to left

    def _tokenize_one(self, text: str, max_length: int = 32) -> list[int]:
        ids = [(ord(c) % max(self.vocab_size - 1, 1)) + 1 for c in text[:max_length]]
        return ids or [1]

    def __call__(
        self,
        text,
        return_tensors: str = "pt",
        padding=False,
        truncation: bool = False,
        max_length: int = 32,
    ):
        # Batch vs single
        if isinstance(text, str):
            ids_list = [self._tokenize_one(text, max_length=max_length if truncation else 32)]
        else:
            ids_list = [
                self._tokenize_one(t, max_length=max_length if truncation else 32)
                for t in text
            ]

        # Padding (respects padding_side like an HF tokenizer)
        if padding:
            target = max(len(ids) for ids in ids_list)
            padded = []
            masks = []
            for ids in ids_list:
                pad_n = target - len(ids)
                if self.padding_side == "left":
                    padded.append([self.pad_token_id] * pad_n + ids)
                    masks.append([0] * pad_n + [1] * len(ids))
                else:
                    padded.append(ids + [self.pad_token_id] * pad_n)
                    masks.append([1] * len(ids) + [0] * pad_n)
            input_ids = torch.tensor(padded, dtype=torch.long)
            attention_mask = torch.tensor(masks, dtype=torch.long)
        else:
            # without padding — all lengths are assumed equal
            input_ids = torch.tensor(ids_list, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

        result = type("Out", (), {})()
        result.input_ids = input_ids
        result.attention_mask = attention_mask
        return result

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            n = token_ids.numel()
        else:
            n = len(token_ids)
        return f"<{n} generated tokens>"

    def apply_chat_template(self, messages, tokenize: bool = False,
                            add_generation_prompt: bool = True, **kwargs) -> str:
        # Real tokenizers accept **kwargs (e.g. enable_thinking). We reflect the toggle
        # with a marker only if it was explicitly passed — the default output (== content) is unchanged.
        content = messages[0]["content"]
        if "enable_thinking" in kwargs:
            content += "<think>" if kwargs["enable_thinking"] else "<think></think>"
        return content


@pytest.fixture
def fake_lm():
    """Fake LM with 4 layers and hidden_dim=64. Enough for smoke."""
    return FakeLM(hidden_dim=64, num_layers=4)


@pytest.fixture
def fake_tokenizer():
    """Fake tokenizer to pair with FakeLM."""
    return FakeTokenizer(vocab_size=100)


@pytest.fixture
def fake_lm_factory():
    """Factory for when different sizes are needed in one test."""
    return FakeLM


@pytest.fixture
def dummy_input():
    """Random input [batch=2, seq=8, hidden=64]."""
    torch.manual_seed(42)
    return torch.randn(2, 8, 64)


@pytest.fixture
def dummy_activations_4_layers():
    """List of 4 activations [batch=2, hidden=64] — for encoder.forward()."""
    torch.manual_seed(42)
    return [torch.randn(2, 64) for _ in range(4)]
