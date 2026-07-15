# CLAUDE.md — daimon framework

Guidance for Claude Code working in this repository.

## What this is

**daimon** trains a thin wrapper (~2% of params) on a **frozen** LLM that reads the base's own
activations, compresses them into **cognitive tokens**, and injects them back through **gated
cross-attention** — giving the model calibrated uncertainty (answer / refuse / look up / clarify). The
base weights never change; gradients flow *through* the frozen base to the wrapper (the base acts as a
proxy loss). The core claim: this latent channel is **not reachable by prompting** (see
`docs/results/qwen-14b/text-vs-latent.md`).

## The seam: a LIBRARY and a FRAMEWORK

The MECHANISM was extracted into its own library — **`meta-attention`** (Apache-2.0, separate
repo, `publish/meta-attention/`): reading activations, encoders, gated cross-attention, the
two-pass pipeline, the checkpoint format, **and the whole C++/ggml + llama.cpp leg**. It knows
nothing about doubt, goals or memory — and a test enforces that (`tests/test_boundary.py`, plus
the library's own).

Why the split is not premature: **two implementations already satisfy the same contract** (torch
and ggml, validated bit-for-bit), and the C++ side never needed to know what a voice is.

```text
meta-attention  the MECHANISM: reader, encoders, injection (gated CA), buffer, pipeline,
                checkpoint format, Injector protocol, cpp/ (ggml + llama.cpp patch)   → nothing

daimon-voices   the VOICES + the `Voice` contract + `watchdog`      → meta-attention
daimon-agent    agentic runtime, native tool use, OpenAI serving    → meta-attention
daimon-loom     training + eval + factory + slice_forward           → meta-attention, voices, agent
daimon-deploy   GGUF sidecar export (uses the lib's format)         → meta-attention
daimon          umbrella: re-exports the mechanism (as DaimonPipeline/DaimonConfig) + the voices
```

**`Injector` (library) vs `Voice` (framework)** — the line that matters. `Injector` says only
*when* the mechanism calls you. `Voice` = `Injector` + this framework's opinions: a LIFECYCLE
(reactive / persistent / accumulating — three disciplines that turned out not to be
interchangeable), a CHECKPOINT contract, and a GAIN fader (voices sum on the residual; each has
its own volume).

Each package is a real pip package with its own `pyproject.toml`.

## Commands

```bash
# install (editable). The library first — everything stands on it.
pip install -e ../publish/meta-attention
pip install -e daimon-voices -e daimon-agent -e daimon-loom
pip install -e daimon-deploy          # optional (GGUF export)
pip install -e . --no-deps            # optional umbrella (`daimon`)

# tests — CPU-only, no GPU/network needed. 239 in the framework + 8 in the library.
python -m pytest tests/ -q
(cd ../publish/meta-attention && python -m pytest tests/ -q)
python -m pytest tests/test_build_universal.py -q   # a single file

# the metaloom CLI (stages share a run.json manifest via --run-dir)
metaloom collect --run-dir runs/my --model-name <hf-id> --dataset mmlu --target-layers late --mcq-direct
metaloom train   --run-dir runs/my --epochs 6
metaloom eval    --run-dir runs/my
metaloom build-universal --model-name <hf-id> --quantization nf4 --suite suite.json --eval --export-gguf
daimon-agent run --run-dir runs/my "…"        # agentic runtime
metadeploy export --run-dir runs/my         # → doubter_sidecar.gguf
```

`metaloom` = `daimon_loom/cli/main.py` dispatcher; stages are `collect|train|eval|build-universal` (+ `cloud`).

## Architecture / key concepts

- **Two-pass injection.** Pass-1 reads activations → cognitive tokens (buffered); Pass-2 injects them via
  gated cross-attention and generates. `DaimonPipeline` orchestrates it; `_run_pass1()` + the
  collector's freeze/unfreeze gate it.
- **`run.json` manifest** is the single source of truth threading the stages (model, resolved layers,
  encoder type, dtype, quantization, chat_template_kwargs). `collect` writes it (freezing the *resolved*
  layer indices + `hidden_dim`/`num_layers`); `train`/`eval`/`deploy` read it via `--run-dir`.
- **Layer presets** are model-agnostic: `target_layers="late"` / `"all"` / `"late_slice"` resolved by the
  framework from the base's `num_layers` (`daimon_loom.cli._common.parse_layers`). Prefer `"late"`.
- **Checkpoint contract** (`.pt`): `{format_version, config, encoder_state, ca_state}`. `Doubter.from_checkpoint`
  rebuilds from `config`. GGUF sidecar = the same tensors as float32 + config in metadata.
- **Runtime knob:** `pipe.set_gain(x)` / `doubter.set_gain(x)` (`META_GAIN` in llama.cpp). **AGC** damps the
  self-amplifying injection over long generation (decay → floor).
- **The Daimon leg** (the `daimon-voices` PACKAGE): ALL injection voices are one family — the
  frozen model's inner advisory voices (Socratic daimonion: counsels, doesn't rule). Doubter (doubt),
  GoalAnchor (goal), Chronographer (memory), ChronoAnchor (the two fused); planned: Reassembler.
  Voices sum on the residual, each with its own `gain` fader. Probes that only listen
  (`watchdog`) are NOT Daimons — they gate.
- **Watchdog** (`daimon_voices.watchdog`): a `ConfidenceProbe` on the cognitive token that GATES actions.
  Its headline job is FIXING injection on long (code) generation: constant injection corrupts long
  output (−3 on ODEX), gating it to the decision window makes it safe (+1) — read-only gating of an
  external action (lookup/refuse/escalate, also +1) is the secondary mode, not the whole point.
- **ChronoAnchor** (`daimon_voices.chrono_anchor` + `daimon_loom.training.ChronoAnchorTrainer`): the
  GoalAnchor and the Chronographer FUSED — the goal is not a second voice on the residual but a
  privileged INPUT of the memory organ (it conditions episode compression). Training recipe, all four
  ingredients measured and load-bearing: **center the goal latent** (else the constant component eats
  the channel), **focus the loss on decision tokens**, **diverse goal phrasings**, **train to the loss
  plateau**. The goal is read from the FROZEN base's own activations — no separate anchor encoder.
- **build-universal** (`daimon_loom.cli.build_universal` + `daimon_loom.data.agentic_mix`): assembles a
  balanced commit+hold diverse mix → collect → `train_stage(agentic_targets=True)` → optional per-axis
  eval + GGUF. A "factory" for a general wrapper over any model.

## Testing conventions

- Tests run on **CPU with a `FakeLM` / `FakeTokenizer`** (`tests/conftest.py`) — no HF download, no GPU.
- Stages are **injectable**: `collect_stage(..., pipeline=, questions=, ground_truths=)`,
  `train_stage(..., pipeline=, samples=)`, `build_universal_stage(..., pipeline=, mix=)`. Use
  `_fake_pipe(fake_lm_factory)` (in `tests/test_cli.py`) to build a fake pipeline.
- `FakeTokenizer` does NOT accept `add_special_tokens` and its `forward` returns hidden states (not
  logits) — code that scores tokens must fall back to `model.lm_head(hidden)` when the output has no
  `.logits` (see `daimon_loom/evaluation/agentic_suite.py`).
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
  `daimon_loom.evaluation.harness.REFUSAL_PHRASES`), NOT via an artificial "UNSURE" MCQ option (that readout
  makes it look inert — a real bug we hit). Save raw generations to catch such mismatches by eye.
- **Don't trust accuracy without the label distribution** — a single-class subset can fake a huge win;
  check macro-F1 / the class balance.

## Where things live

- `../publish/meta-attention/` — **the library** (separate repo, Apache-2.0): `pipeline.py`,
  `injection.py` (gated CA), `reader.py` (hooks), `buffer.py`, `encoders/`, `checkpoint.py`
  (the format contract), `injector.py` (the protocol), `cpp/` (ggml + the llama.cpp patch).
- `daimon-voices/daimon_voices/voice.py` — the `Voice` contract (Injector + lifecycle + checkpoint
  + gain); `watchdog.py` — the read-side probe that GATES (not a voice).
- `daimon-voices/daimon_voices/` — the Daimon leg: `chrono_anchor.py` (the ONE organ: goal-
  conditioned episodic memory + the agentic lifecycle — `set_goal` + `step_hooks`), `chronographer.py`
  (rolling episode bank), `doubter.py`, `goal_anchor.py` (persistent
  anchor + trigger-gated CA), `config.py` (DoubterConfig, GoalAnchorConfig).
- `daimon-loom/daimon_loom/` — `cli/` (`collect.py`, `train.py`, `eval.py`, `build_universal.py`, `_common.py`),
  `training/` (`trainer.py`, `losses.py`), `data/` (`agentic_mix.py`, `dataset.py`, `splits.py`),
  `evaluation/` (`harness.py`, `agentic_suite.py`, `benchmark.py`).
- `daimon-deploy/daimon_deploy/export.py` — `export_from_run_dir` / `export_sidecar` (GGUF).
- `docs-site/index.html` — the static docs (EN+RU). Published on Codeberg Pages from the **`pages`** branch
  (site content at root); `main` holds it under `docs-site/`. Assets in `docs-site/assets/*.svg`.
- `CHANGELOG.md` — release notes (currently v0.3.1).

## Conventions

- Match the surrounding code's style (dense, comment-light where obvious; the CLI stages carry `# F#:`
  fix-notes for silent-failure guards — keep them).
- Don't commit weights/data/`.pt`/`out/` (gitignored). Report results under `docs/results/<model>/`.
- Publishing: framework → Codeberg (`codeberg.org/imperius/daimon`) + GitHub mirror
  (`github.com/artem-x-meta/daimon`) — both are remotes of the LIVE publish clone at
  `publish/daimon-framework/` in the parent repo (its own git repo, gitignored by the parent;
  push via native PowerShell git). Per-model wrappers → HuggingFace
  (`HF_HUB_ENABLE_HF_TRANSFER=1`, per-file for big files).
- **This folder is the DEV tree; `publish/daimon-framework/` is the published branch.**
  Sync publishable changes dev → publish clone → commit → push both remotes. Unvalidated R&D
  (e.g. `voices/goal_anchor.py`, `data/drift_sessions.py`) stays dev-only until validated.
