# Changelog

## Unreleased

### Added
- **GoalAnchor is now published** (early-stage). The goal-drift voice — a PERSISTENT latent anchor
  encoded once from the goal/spec text, trigger-gated re-injected across generations — ships in
  `meta-daimon` (was dev-only). Measured on Qwen2.5-14B agentic coding: diverse training over
  constraint families gives **constraint defense +19pp** (in-domain, where the base drifts) and
  **transfer without quality loss** on unseen constraint families (a narrow anchor cost −19pp).
  Anchor on HF: `Imperius/meta-qwen-14b-goalanchor`. Data pipeline in `meta_loom.data`
  (`code_spec_sessions`, `drift_sessions`).
- **`metaloom build-anchor` — the GoalAnchor factory.** Mirrors `build-universal` for the
  behavior modifier: mine self-distilled pairs over constraint families → train a GoalAnchor
  → save (+ optional GGUF). `slice_train` optional (default on: fits a 14B anchor on a small
  GPU; off = full two-pass Trainer). Also a core fix: the anchor's decision hook installs only
  for triggered modes, so an always-anchor attaches on a model of any depth.
- **GGUF export for GoalAnchor** (`meta_deploy.export_anchor_sidecar`; `metadeploy export` autodetects
  `kind=goal_anchor`). Sidecar carries `meta_spider.kind` + trigger metadata (trigger / trigger_k /
  decision_layer). NB: the C++ llama.cpp runtime for the anchor's lifecycle (static anchor +
  trigger gating) is not implemented yet — the GGUF is export-ready, the hook is a TODO.

### Changed
- **Package split: the Meta-Daimon leg is now its own package.** `meta-core` keeps the
  abstract meta-attention MECHANISM (pipeline, hooks, encoders, gated CA, the `Modifier`
  contract, watchdog probe, checkpoint contract); the concrete injection modifiers — the
  VOICES (Doubter, GoalAnchor + `DoubterConfig`/`GoalAnchorConfig`) — moved to the new
  `meta-daimon` package (named after the Socratic daimonion: counsels, doesn't rule).
  Back-compat: `from meta_core import Doubter` / `meta_core.modifiers.Doubter` are lazily
  forwarded to `meta_daimon` (PEP 562), the `meta_spider` umbrella re-exports everything.
  Install order: core → daimon → agent → loom. Checkpoints unaffected (configs are saved
  as plain dicts).

### Fixed

- `check_gsm8k_answer` no longer misparses phrasal answers: the number regex captured a
  sentence-ending period ("The answer is 5." -> "5." != "5") and a lone comma matched as
  a "number" — both silently deflated the oracle. Now `-?\d[\d,]*(?:\.\d+)?` (+ trailing-dot
  strip in the `####` branch). Found by a second agent instance working on the publish tree.
- `REFUSAL_PHRASES` now include the factory mix's trained refusal target (`agentic_mix.REFUSE`:
  "don't have enough information", "can't/cannot provide") — the detector previously missed the
  diverse wrapper's OWN refusals (caught by eye in saved generations; offline regrade confirmed).
- **Refusal detection is now commit-first (opening-sentence rule).** `harness.classify_action`
  and `meta_agent.action.looks_like_refusal` count a refusal only when the phrase appears in the
  OPENING sentence; an answer followed by trailing doubt ("F. 200 m/s … I'm not confident") is a
  commit. Phrase-anywhere matching inflated refusal rates on pre-EOS-fix checkpoints that ramble
  doubt phrases after answering (caught by eye in the v0.3.1 re-measure generations: 12/12
  "refusals" on the memory axis were actually answered). Corrections stay phrase-anywhere by
  design (the trained template follows a first answer mid-text).

### Added
- **`GoalAnchor` — the goal-drift Watchdog leg (behavior modifier).** Port of the archived
  `watchdog_v2_llama1b` (v2.1, validated on Llama-3.2-1B: the latent goal anchor matches gold
  text reminders on the forbid family WITHOUT the goal in the prompt, at ~30% fewer
  interventions; within-family generalization 3/3 WIN). The GOAL text is encoded once into a
  static anchor; during generation trigger-gated cross-attention re-injects it. Lifecycle
  contrast vs the Doubter: the anchor PERSISTS across prompts (`on_pre_forward` resets only
  the trigger). Triggers: `always` / `fixed` (every K tokens) / `learnable` (MLP probe on the
  decision layer + cooldown). Checkpoint kind `goal_anchor` (v1.1 contract, strict layer
  verify, learnable-trigger state included). Not to be confused with the
  `meta_core.watchdog.Watchdog` confidence sensor (whose probe gates pointwise injection /
  external actions). Known measured limits: no transfer to an
  unseen drift FAMILY; training pipeline port (meta-loom stages) is the next step.
  **Agentic mode `trigger="agent_step"` (the project's aim):** injection is OFF by default
  and armed only for the agent-loop DECISION window — `MetaAgent(step_hooks=[anchor])` calls
  `on_step_start`/`on_step_end` around each `policy.act` (decide-then-detach). NB: this regime is a HYPOTHESIS by
  analogy with the Doubter's ODEX finding — v2 itself validated sparse in-generation pulses
  (fixed-K); comparing the two regimes is part of the agentic bench, not settled. Agentic validation needs a
  bigger base (Qwen-14B / Granite-8B) — 1B is below the agentic capability floor; plan in
  `docs/project_notes/goal-anchor-agentic-plan.md`.
  Tests `tests/test_goal_anchor.py` (10). Suite 164 → 174.

## v0.3.1 — arbiter-audit fixes (correctness + honest-metrics hardening)

An external-arbiter audit of the whole framework; every confirmed finding fixed.

### Fixed
- **Inference ran with dropout ON.** `.eval()` was only ever called inside the Trainer; the pure
  inference path (`Doubter.from_checkpoint` → `attach` → generate/eval) left the CA modules in
  training mode, so `attn_dropout(0.1)` fired on every token of every CA layer — stochastic
  injection even under greedy decoding. `on_attach` now ends with `set_inference_mode()` (the
  Trainer still flips `.train()`/`.eval()` itself). **Re-measure published numbers** — expect the
  wrapper's metrics to improve slightly (noise removed); conclusions should hold.
- **Silent partial checkpoint load.** A checkpoint attached to a pipeline with different layers
  loaded only the intersecting CA modules and left the rest randomly initialized — no warning.
  Now: checkpoints (v1.1) carry `target_layers`/`cross_attn_layers`, attach verifies them, and a
  `ca_state` key mismatch raises instead of half-loading.
- **No stop supervision in training targets.** Targets carried no EOS, and the `labels[labels ==
  pad_id] = -100` mask (with the common pad==eos setup) erased even a real EOS — the wrapper never
  learned to STOP after the refusal phrase (observed as rambling continuations). Targets now end
  with the tokenizer's EOS; padding is masked BY POSITION via attention_mask (value-mask only as a
  maskless fallback). Same fix in the slice-collector path and `make_labels_with_prompt_mask`.
- **Val loss was incomparable across epochs.** With `correction_ratio>0`, val targets were
  re-sampled every epoch from the shared RNG (noisy early-stop/best-model, shifted train shuffle).
  Val targets are now built once, from a dedicated RNG.
- **MCQ oracle over-counted verbose answers.** A single-letter truth matched `\bB\b` anywhere
  ("Between B and C, I'd pick C" counted as B). `check_answer_correctness` now EXTRACTS the chosen
  letter (`extract_mcq_letter`: explicit "answer is X" → leading letter → the only distinct
  letter; ambiguous → not correct). The harness/collector default checkers now delegate to the
  same function (three inconsistent checkers → one).
- **Correction metrics fired on ordinary speech.** `CORRECTION_PHRASES` tightened to the trained
  template ("wait, the correct…"); bare "actually"/"wait"/"let me think" no longer count.
- Trailing gradient-accumulation window is flushed (was silently wiped by next epoch's
  `zero_grad`); fully-truncated targets are counted and warned about (were silently NaN-skipped);
  out-of-range `target_layers` raise instead of a silent skip; `ReflexionBuffer.expand_batch`
  raises on a filled-buffer/batch mismatch; collector freeze/unfreeze wrapped in try/finally.

### Added
- **`--abstain-affordance` (collect)** — the fair-baseline lesson as a flag: appends an explicit
  "you may say you are not confident" instruction inside the user turn, so BOTH eval arms (base
  and Doubter) get the same refusal affordance and the refusal delta is not a prompt artifact.
- **`--tool-call-format auto|qwen|granite|llama` (build-universal / agentic_mix)** — the factory's
  tool-call TARGETS previously hardcoded Qwen's `<tool_call>` syntax while claiming
  model-agnosticism; the format is now detected from the chat template (with a loud fallback
  warning) or set explicitly.
- Honest-generality caveat documented: suite v1 shares sources (When2Call/PopQA/SQuAD2) with the
  diverse mix — a source-disjoint suite v2 is the real generality test.
- Tests `tests/test_arbiter_fixes.py` (14). Suite 153 → 164.

## v0.3.0 — the universal-Doubter factory (`build-universal`)

### Added
- **`metaloom build-universal --model-name N`** — one command to build a *general* uncertainty Doubter
  for any model: assembles a **balanced diverse agentic mix** (commit + hold across the decision space),
  runs collect → train model-agnostically (`--target-layers late`), and writes a publishable run-dir.
  Validated on Qwen2.5-14B: the diverse-trained wrapper was the only arm with no axis collapse on the
  6-axis suite (floor 0.467, commit preserved) — `docs/results/qwen-14b/diverse-train-balanced.md`.
- `meta_loom.data.agentic_mix` — the reusable diverse-mix builder (`build_training_mix(tokenizer, …)`):
  *call* from When2Call `train_pref.chosen_response` (NB: the SFT split has no tool calls), *memory*
  from PopQA-popular + SQuAD-answerable, *abstain/clarify* from When2Call SFT, *lookup* from PopQA
  long-tail, *unknown* from SQuAD-unanswerable. `--suite` excludes the held-out suite's questions
  (leakage guard). `targets_from_samples()` feeds the Trainer's `targets_by_sample`.
- `train_stage(agentic_targets=True, init_from=…)` — explicit multi-action routing targets, and
  continuing from an existing wrapper (e.g. a QA Doubter → agentic).
- `meta_loom.evaluation.agentic_suite` — per-axis log-prob suite eval (`compare_base_vs_doubter`):
  reports per-axis accuracy + the balance aggregates **floor** (worst axis) and **commit_mean**
  (call + memory). Wired into `build-universal --eval`. (Action axes exact; knowledge axes via
  log-prob are approximate — a generation+judge eval is more faithful there.)
- `build-universal --export-gguf` — also emit the llama.cpp GGUF sidecar (via meta-deploy, if installed).
- Tests `tests/test_build_universal.py` (8). Suite 145 → 153.

### Why
The recipe that produced the balanced general Doubter was a one-off vast/Kaggle script; this makes it a
turnkey framework feature — a "factory" for universal uncertainty wrappers over arbitrary base models.

## v0.2.1 — first-class train/val/test split + leakage guard

### Added
- `meta_loom.data.split_samples(samples, train, val, test, verify=True)`: the canonical holdout split
  (the list is shuffled at collect time; sequential slices are disjoint). With `verify=True` it ASSERTS
  no question (`input_text`) appears in two splits — raises `ValueError` on leakage. `metaloom train`
  and `metaloom eval` now go through it (previously each hand-sliced the samples with no guard).
- `meta_loom.data.assert_disjoint_from(samples, holdout)`: guard for the cross-run / cross-dataset case
  (e.g. train on full-mmlu, eval on mmlu_hard from a different collect — different index spaces).
- Tests `tests/test_splits.py` (6). Suite 132 → 138.

### Why
Classic DL hygiene the toolkit was missing as an explicit safeguard: a frozen base does not prevent the
wrapper from memorising per-question correctness, so train/test must be provably disjoint.

## v0.2.0 — the uncertainty potentiometer

**Headline: a runtime gain knob on the latent injection.** The Doubter's cross-attention injection is
now a continuous, content-orthogonal *control knob*, not a fixed on/off. This is the first "fader" of the
mixing-console vision (each behaviour modifier = a channel with its own gain on the shared residual stream).

### Added
- `BottleneckCrossAttention.gain` (default `1.0`) + `set_gain(g)`: a static multiplier on the injection,
  applied across all return paths (plain / AGC-decay / AGC-measure). `gain=1.0` = trained baseline,
  `>1` amplify (more doubt → more refusal), `<1` attenuate, `<0` invert (toward confidence). Composes with
  the trained per-layer gates and with AGC. Not trained — set at inference.
- `Doubter.set_gain(g)` / `get_gain()`: one knob over every CA layer of the wrapper.
- `MetaSpiderPipeline.set_gain(g)`: convenience that sets the gain on every attached modifier that supports
  it (the mixing console — dial all behaviour channels at once).
- Test `test_bottleneck_ca_gain_potentiometer`: gain scales the injection linearly (0 → none, 2 → double,
  −1 → invert).

### Validated (on `google/gemma-4-12B-it`)
- Monotonic refusal dial: `gain` 0 → 1.5 moves refusal 2% → 51% (see
  `docs/results/gemma-4-12b/gain-potentiometer.md`).
- Stability ceiling ~1.5–2.0: beyond it the positive-feedback injection runs away (degeneration 79% at
  gain=3) — combine the static gain with AGC near the ceiling.
- Coverage knob, not free calibration: selective accuracy does not improve as gain rises (the knob trades
  coverage for precision along one curve).

### Backward compatibility
- `gain` defaults to `1.0` → behaviour unchanged for existing checkpoints/code. 132/132 tests green.

### Notes
- Kaggle payload copies under `lab/experiments/*/kaggle/payload/meta_core/` are vendored snapshots and are
  NOT auto-synced; re-vendor to use the framework knob from kernels (the gain-sweep kernel used a runtime
  forward-hook against the old payload).

## v0.0.1 — initial split
- Meta-Spider split into `meta_core` / `meta_loom` / `meta_agent` / `meta_deploy` (+ `meta_spider` compat shim).
- AGC (Automatic Gain Control) regulator on the injection (decay/measure modes) — damps the doubt-accumulation
  runaway.
