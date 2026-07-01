"""Diverse agentic training mix — the data side of the "universal Doubter factory".

A *balanced* mix of commit and hold decisions across the agentic decision space, so one wrapper
generalizes instead of over-fitting into caution. Validated on Qwen2.5-14B (vast, 30.06.2026):
the diverse-trained Doubter was the only arm with no axis collapse (floor 0.467, commit preserved).
See `docs/results/qwen-14b/diverse-train-balanced.md`.

Axes → source → target (all disjoint from any held-out eval suite via `exclude_questions`):
  commit:  call    — When2Call `train_pref.chosen_response` <TOOLCALL> → native tool call
           memory  — PopQA high-popularity + SQuAD2-train answerable → direct answer (gold)
  hold:    abstain/clarify — When2Call `train_sft`
           lookup  — PopQA long-tail → search tool call
           unknown — SQuAD2-train unanswerable → refuse

Model-agnostic: takes a tokenizer (applies its own chat template, counts its own tokens). Returns
`(prompts, specs)` where each spec is a JSON string {"target", "label"} — the collector stores it as
`ground_truth`, and `targets_from_samples()` turns it back into the Trainer's `targets_by_sample`.

NB (measured): the positive *call* examples live ONLY in `train_pref.chosen_response`, NOT in the SFT
split (which is abstain/clarify only) — missing this silently zeroes the commit side.
"""
from __future__ import annotations

import json
import random
import re
from typing import Optional

# the six decision axes and which side (commit = act, hold = withhold/redirect) each belongs to
AXES = ("call", "memory", "abstain", "clarify", "lookup", "unknown")
COMMIT_LABELS = ("call", "memory")

SYSTEM_DEFAULT = (
    "You are a helpful AI assistant with optional tools. Answer directly only if you are sure; "
    "use a tool to look it up if unsure of a fact; ask to clarify if ambiguous; decline if unanswerable."
)
SEARCH_TOOL = ['{"name": "search", "description": "Look up a fact about an entity.", "parameters": '
               '{"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}']
REFUSE = "I don't have enough information to answer that reliably."
_TOOLCALL_RE = re.compile(r"<TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>", re.S)
# When2Call schemas use python-style type names; map to JSON-schema for apply_chat_template(tools=…)
_TYPEMAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _norm(s) -> str:
    return " ".join(str(s).split()).lower().strip()


def fix_types(o):
    if isinstance(o, dict):
        return {k: (_TYPEMAP.get(v, v) if k == "type" and isinstance(v, str) else fix_types(v))
                for k, v in o.items()}
    if isinstance(o, list):
        return [fix_types(x) for x in o]
    return o


def to_tools(ts):
    out = []
    for t in (ts or []):
        try:
            out.append({"type": "function", "function": fix_types(json.loads(t) if isinstance(t, str) else t)})
        except Exception:
            pass
    return out


# Native tool-call TARGET formats per model family. The prompt side is model-agnostic
# (the tokenizer renders its own template), but the target the wrapper is trained to EMIT
# must match the family's native syntax — training a Llama wrapper to emit Qwen's
# `<tool_call>` tags would produce calls its own parser never fires on.
TOOL_CALL_FORMATS = {
    # Qwen2.5 / Hermes-style
    "qwen": '<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>',
    # Granite-3.x
    "granite": '<|tool_call|>[{{"name": "{name}", "arguments": {args}}}]',
    # Llama-3.x (json tool calling; uses "parameters", no wrapper tags)
    "llama": '{{"name": "{name}", "parameters": {args}}}',
}


def detect_tool_call_format(tokenizer) -> str:
    """Best-effort detection of the family's native tool-call syntax from the chat template.

    Returns a key of TOOL_CALL_FORMATS. Unknown template → "qwen" with a LOUD warning
    (the historical behavior, correct for Qwen/Hermes-style models only).
    """
    tpl = getattr(tokenizer, "chat_template", None) or ""
    if "<tool_call>" in tpl:
        return "qwen"
    if "<|tool_call|>" in tpl:
        return "granite"
    if "<|python_tag|>" in tpl or '"parameters"' in tpl:
        return "llama"
    print("  [warn] agentic_mix: could not detect the model's native tool-call syntax from "
          "its chat template — falling back to the Qwen '<tool_call>' format. If the target "
          "model is not Qwen/Hermes-style, pass tool_call_format= explicitly "
          "(one of: " + ", ".join(sorted(TOOL_CALL_FORMATS)) + ").", flush=True)
    return "qwen"


def tool_call_text(name: str, args, fmt: str = "qwen") -> str:
    """Native tool call (what the model emits) for a target string, in the given family format."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    template = TOOL_CALL_FORMATS.get(fmt)
    if template is None:
        raise ValueError(f"unknown tool_call_format {fmt!r}; one of {sorted(TOOL_CALL_FORMATS)}")
    return template.format(name=name, args=json.dumps(args))


def make_example(tokenizer, user: str, tools, target: str, label: str, *, system: str) -> dict:
    """One mix item: chat-templated prompt + spec {target,label}. tools=None → no tools in the prompt."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=to_tools(tools) if tools else None, add_generation_prompt=True, tokenize=False)
    return {"prompt": prompt, "spec": json.dumps({"target": target, "label": label})}


def targets_from_samples(samples) -> list[tuple[str, str]]:
    """Collected samples → Trainer `targets_by_sample` = [(target_text, label), …] from ground_truth spec."""
    out = []
    for s in samples:
        d = json.loads(s.ground_truth)
        out.append((d["target"], d["label"]))
    return out


def build_training_mix(
    tokenizer,
    *,
    per_class: int = 70,
    cap_tok: int = 320,
    exclude_questions: Optional[set] = None,
    seed: int = 0,
    system: str = SYSTEM_DEFAULT,
    tool_call_format: str = "auto",
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """Build the balanced diverse mix. Returns (prompts, specs). Requires `datasets` (HF) + network.

    `exclude_questions` — normalized test-suite questions to drop (leakage guard). `per_class` items
    per class-source (memory pulls from two sources → ~2× memory). Items over `cap_tok` tokens skipped.
    `tool_call_format` — the native syntax of the tool-call TARGETS ("auto" = detect from the
    tokenizer's chat template; see TOOL_CALL_FORMATS).

    Honest caveat on "generality": the held-out suite v1 draws from the SAME three sources
    (When2Call / PopQA / SQuAD2) as this mix, disjoint only by exact question text — suite scores
    partially reflect template familiarity, not pure transfer. A source-disjoint suite (v2) is the
    real generality test.
    """
    from datasets import load_dataset

    if tool_call_format == "auto":
        tool_call_format = detect_tool_call_format(tokenizer)
    if verbose:
        print(f"  tool-call target format: {tool_call_format}", flush=True)

    exclude = exclude_questions or set()
    rng = random.Random(seed)
    pool: list[dict] = []

    def _len(item) -> int:
        return len(tokenizer(item["prompt"]).input_ids)

    # ── When2Call: call ← train_pref.chosen_response <TOOLCALL>; clarify/abstain ← train_sft ──
    cnt = {"call": 0, "clarify": 0, "abstain": 0}
    pref = load_dataset("nvidia/When2Call", "train_pref")["train"]
    for r in pref:
        if cnt["call"] >= per_class:
            break
        tools = r["tools"]
        msgs = r["messages"]
        if isinstance(tools, str):
            tools = json.loads(tools)
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        user = next((x["content"] for x in msgs if x["role"] == "user"), None)
        ch = r.get("chosen_response")
        if isinstance(ch, str):
            try:
                ch = json.loads(ch)
            except Exception:
                ch = {"content": ch}
        c = (ch or {}).get("content") or ""
        mt = _TOOLCALL_RE.search(c)
        if not user or not mt or _norm(user) in exclude:
            continue
        try:
            calls = json.loads(mt.group(1))
            tgt = tool_call_text(calls[0]["name"], calls[0].get("arguments", {}), fmt=tool_call_format)
        except Exception:
            continue
        it = make_example(tokenizer, user, tools, tgt, "call", system=system)
        if _len(it) > cap_tok:
            continue
        pool.append(it)
        cnt["call"] += 1
    w2c = load_dataset("nvidia/When2Call", "train_sft")["train"]
    for r in w2c:
        if cnt["clarify"] >= per_class and cnt["abstain"] >= per_class:
            break
        tools = r["tools"]
        msgs = r["messages"]
        if isinstance(tools, str):
            tools = json.loads(tools)
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        user = next((x["content"] for x in msgs if x["role"] == "user"), None)
        asst = next((x for x in msgs if x["role"] == "assistant"), None)
        if not user or not asst or _norm(user) in exclude:
            continue
        content = asst.get("content") or ""
        lab = "clarify" if "?" in content[-60:] else "abstain"
        tgt = content or REFUSE
        if cnt[lab] >= per_class:
            continue
        it = make_example(tokenizer, user, tools, tgt, lab, system=system)
        if _len(it) > cap_tok:
            continue
        pool.append(it)
        cnt[lab] += 1
    if verbose:
        print(f"  When2Call: {cnt}", flush=True)

    # ── PopQA: memory (high-popularity, gold) + lookup (long-tail → search tool call) ──
    pop = load_dataset("akariasai/PopQA", split="test")
    pc = {"memory": 0, "lookup": 0}
    for r in pop:
        user = r["question"]
        if _norm(user) in exclude:
            continue
        try:
            gold = json.loads(r["possible_answers"])[0]
        except Exception:
            gold = r.get("obj") or ""
        sp = r.get("s_pop") or 0
        if sp >= 50000 and pc["memory"] < per_class:
            it = make_example(tokenizer, user, None, f"{gold}", "memory", system=system)
        elif sp <= 100 and pc["lookup"] < per_class:
            it = make_example(tokenizer, user, SEARCH_TOOL,
                              tool_call_text("search", {"query": user[:80]}, fmt=tool_call_format), "lookup", system=system)
        else:
            continue
        if _len(it) > cap_tok:
            continue
        pool.append(it)
        pc[json.loads(it["spec"])["label"]] += 1
        if all(pc[k] >= per_class for k in pc):
            break
    if verbose:
        print(f"  PopQA: {pc}", flush=True)

    # ── SQuAD2-train: memory (answerable, gold) + unknown (unanswerable → refuse) ──
    sq = load_dataset("rajpurkar/squad_v2", split="train")
    sc = {"memory": 0, "unknown": 0}
    for r in sq:
        q = f"Context: {r['context'][:500]}\n\nQuestion: {r['question']}"
        if _norm(r["question"]) in exclude:
            continue
        if r["answers"]["text"] and sc["memory"] < per_class:
            it = make_example(tokenizer, q, None, r["answers"]["text"][0], "memory", system=system)
        elif not r["answers"]["text"] and sc["unknown"] < per_class:
            it = make_example(tokenizer, q, None, REFUSE, "unknown", system=system)
        else:
            continue
        if _len(it) > cap_tok:
            continue
        pool.append(it)
        sc[json.loads(it["spec"])["label"]] += 1
        if all(sc[k] >= per_class for k in sc):
            break
    if verbose:
        print(f"  SQuAD2-train: {sc}", flush=True)

    rng.shuffle(pool)
    import collections
    labs = collections.Counter(json.loads(x["spec"])["label"] for x in pool)
    commit = sum(labs[k] for k in COMMIT_LABELS)
    if verbose:
        print(f"  DIVERSE MIX: {len(pool)} | labels={dict(labs)} | "
              f"commit={commit} hold={len(pool) - commit}", flush=True)
    if not pool:
        raise RuntimeError("empty diverse mix — check dataset availability / exclude set")
    return [x["prompt"] for x in pool], [x["spec"] for x in pool]
