# CLAUDE.md — meta-spider framework

Guidance for Claude Code working in this repository.

## What this is

**meta-spider** trains a thin wrapper (~2% of params) on a **frozen** LLM that reads the base's own
activations, compresses them into **cognitive tokens**, and injects them back through **gated
cross-attention** — giving the model calibrated uncertainty (answer / refuse / look up / clarify). The
base weights never change; gradients flow *through* the frozen base to the wrapper (the base acts as a
proxy loss). The core claim: this latent channel is **not reachable by prompting** (see
`docs/results/qwen-14b/text-vs-latent.md`).

## Packages (five, one job each — dependency graph)

```text
meta-core   the meta-attention MECHANISM (frozen base, hooks, encoders, gated CA, Modifier
            contract, checkpoint contract) → depends on NOTHING.
meta-daimon the VOICES (injection modifiers: Doubter, GoalAnchor + their configs) → meta-core
meta-agent  agentic runtime + native tool-use            → meta-core
meta-loom   training + eval + the build-universal factory → meta-core + meta-daimon + meta-agent
meta-deploy export to a GGUF sidecar for llama.cpp (CPU)  → meta-core
meta_spider umbrella shim: `from meta_spider import *` re-exports core + daimon + loom (back-compat)
```

Each is a real pip package with its own `pyproject.toml` under `meta-spider-framework/<pkg>/`.

## Commands

```bash
# install (editable, from this folder). Order matters: core → daimon → agent → loom → umbrella.
pip install -e meta-core -e meta-daimon -e meta-agent -e meta-loom
pip install -e meta-deploy          # optional (GGUF export)
pip install -e . --no-deps          # optional umbrella (meta_spider shim)

# tests — CPU-only, no GPU/network needed (run from framework root). ~172 pass.
python -m pytest tests/ -q
python -m pytest tests/test_build_universal.py -q   # a single file

# the metaloom CLI (stages share a run.json manifest via --run-dir)
metaloom collect --run-dir runs/my --model-name <hf-id> --dataset mmlu --target-layers late --mcq-direct
metaloom train   --run-dir runs/my --epochs 6
metaloom eval    --run-dir runs/my
metaloom build-universal --model-name <hf-id> --quantization nf4 --suite suite.json --eval --export-gguf
meta-agent run --run-dir runs/my "…"        # agentic runtime
metadeploy export --run-dir runs/my         # → doubter_sidecar.gguf
```

`metaloom` = `meta_loom/cli/main.py` dispatcher; stages are `collect|train|eval|build-universal` (+ `cloud`).

## Architecture / key concepts

- **Two-pass injection.** Pass-1 reads activations → cognitive tokens (buffered); Pass-2 injects them via
  gated cross-attention and generates. `MetaSpiderPipeline` orchestrates it; `_run_pass1()` + the
  collector's freeze/unfreeze gate it.
- **`run.json` manifest** is the single source of truth threading the stages (model, resolved layers,
  encoder type, dtype, quantization, chat_template_kwargs). `collect` writes it (freezing the *resolved*
  layer indices + `hidden_dim`/`num_layers`); `train`/`eval`/`deploy` read it via `--run-dir`.
- **Layer presets** are model-agnostic: `target_layers="late"` / `"all"` / `"late_slice"` resolved by the
  framework from the base's `num_layers` (`meta_loom.cli._common.parse_layers`). Prefer `"late"`.
- **Checkpoint contract** (`.pt`): `{format_version, config, encoder_state, ca_state}`. `Doubter.from_checkpoint`
  rebuilds from `config`. GGUF sidecar = the same tensors as float32 + config in metadata.
- **Runtime knob:** `pipe.set_gain(x)` / `doubter.set_gain(x)` (`META_GAIN` in llama.cpp). **AGC** damps the
  self-amplifying injection over long generation (decay → floor).
- **The Meta-Daimon leg** (the `meta-daimon` PACKAGE): ALL injection modifiers are one family — the
  frozen model's inner advisory voices (Socratic daimonion: counsels, doesn't rule). Doubter (doubt),
  GoalAnchor (goal; early-stage, published on HF), planned: Reassembler, Chronographer. Voices sum on the residual, each
  with its own `gain` fader. `meta_core.modifiers` keeps only the `Modifier` CONTRACT + lazy
  back-compat shims (`from meta_core import Doubter` still works). Probes that only listen
  (`watchdog`) are NOT Meta-Daimons — they gate.
- **Watchdog** (`meta_core.watchdog`): a `ConfidenceProbe` on the cognitive token that GATES actions.
  Its headline job is FIXING injection on long (code) generation: constant injection corrupts long
  output (−3 on ODEX), gating it to the decision window makes it safe (+1) — read-only gating of an
  external action (lookup/refuse/escalate, also +1) is the secondary mode, not the whole point.
- **build-universal** (`meta_loom.cli.build_universal` + `meta_loom.data.agentic_mix`): assembles a
  balanced commit+hold diverse mix → collect → `train_stage(agentic_targets=True)` → optional per-axis
  eval + GGUF. A "factory" for a general wrapper over any model.

## Testing conventions

- Tests run on **CPU with a `FakeLM` / `FakeTokenizer`** (`tests/conftest.py`) — no HF download, no GPU.
- Stages are **injectable**: `collect_stage(..., pipeline=, questions=, ground_truths=)`,
  `train_stage(..., pipeline=, samples=)`, `build_universal_stage(..., pipeline=, mix=)`. Use
  `_fake_pipe(fake_lm_factory)` (in `tests/test_cli.py`) to build a fake pipeline.
- `FakeTokenizer` does NOT accept `add_special_tokens` and its `forward` returns hidden states (not
  logits) — code that scores tokens must fall back to `model.lm_head(hidden)` when the output has no
  `.logits` (see `meta_loom/evaluation/agentic_suite.py`).
- Keep new work green: `pytest tests/` before finishing.

## Gotchas (measured, will bite)

- **torch ≥ 2.5 required** — `transformers` (5.11+) calls `nn.Module.set_submodule` (added in 2.5). On an
  older image: `pip install torch==2.5.1` first.
- **Thinking models → always `--mcq-direct`** (Qwen / Gemma-it / Granite). Otherwise the model opens
  `<think>`, never reaches the answer on the short Pass-1, the oracle flag stays 0, and the Doubter
  collapses into permanent refusal. Emits both `enable_thinking=false` + `thinking=false` (model-agnostic).
- **Wrappers are model-specific** — calibrated to one base's activation distribution; a wrapper does not
  transfer to another model or even another fine-tune. Retrain (or `build-universal`) per model.
- **Gemma in fp16 → NaN logits** (softcap/norms overflow) → silent empty output. `_common.build_meta_config`
  auto-switches Gemma to bf16; Gemma-4 also needs `attn_implementation="eager"`.
- **nf4 needs compute capability ≥ 7.5** (P100/V100 7.0 kill it → use T4/Ampere+/3090).
- **Read the latent signal through its trained channel** — the wrapper *generates* a refusal phrase
  ("I'm not confident enough to answer"), so detect refusal in generation (see
  `meta_loom.evaluation.harness.REFUSAL_PHRASES`), NOT via an artificial "UNSURE" MCQ option (that readout
  makes it look inert — a real bug we hit). Save raw generations to catch such mismatches by eye.
- **Don't trust accuracy without the label distribution** — a single-class subset can fake a huge win;
  check macro-F1 / the class balance.

## Where things live

- `meta-core/meta_core/` — `pipeline.py`, `cross_attention.py`, `buffer.py`, `hooks.py`, `watchdog.py`,
  `encoders/` (selective / multi_token / transformer), `model_utils.py`, `slice_forward.py`;
  `modifiers/` — the `Modifier` contract only (voices moved out).
- `meta-daimon/meta_daimon/` — the Meta-Daimon leg: `doubter.py`, `goal_anchor.py` (persistent
  anchor + trigger-gated CA), `config.py` (DoubterConfig, GoalAnchorConfig).
- `meta-loom/meta_loom/` — `cli/` (`collect.py`, `train.py`, `eval.py`, `build_universal.py`, `_common.py`),
  `training/` (`trainer.py`, `losses.py`), `data/` (`agentic_mix.py`, `dataset.py`, `splits.py`),
  `evaluation/` (`harness.py`, `agentic_suite.py`, `benchmark.py`).
- `meta-deploy/meta_deploy/export.py` — `export_from_run_dir` / `export_sidecar` (GGUF).
- `docs-site/index.html` — the static docs (EN+RU). Published on Codeberg Pages from the **`pages`** branch
  (site content at root); `main` holds it under `docs-site/`. Assets in `docs-site/assets/*.svg`.
- `CHANGELOG.md` — release notes (currently v0.3.1).

## Conventions

- Match the surrounding code's style (dense, comment-light where obvious; the CLI stages carry `# F#:`
  fix-notes for silent-failure guards — keep them).
- Don't commit weights/data/`.pt`/`out/` (gitignored). Report results under `docs/results/<model>/`.
- Publishing: framework → Codeberg (`codeberg.org/imperius/meta-spider`) + GitHub mirror
  (`github.com/artem-x-meta/meta-spider`) — both are remotes of the LIVE publish clone at
  `publish/meta-spider-framework/` in the parent repo (its own git repo, gitignored by the parent;
  push via native PowerShell git). Per-model wrappers → HuggingFace
  (`HF_HUB_ENABLE_HF_TRANSFER=1`, per-file for big files).
- **This folder is the DEV tree; `publish/meta-spider-framework/` is the published branch.**
  Sync publishable changes dev → publish clone → commit → push both remotes. Unvalidated R&D
  (e.g. `modifiers/goal_anchor.py`, `data/drift_sessions.py`) stays dev-only until validated.
