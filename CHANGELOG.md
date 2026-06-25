# Changelog

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
