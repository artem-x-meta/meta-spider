"""Per-axis agentic-suite eval (log-prob MCQ) for the universal-Doubter factory.

Given a suite of items {question, tools, options:{direct,tool_call,clarify,refuse}, correct, axis},
score each behavior option by byte-normalized log-prob and take argmax == correct. Reports per-axis
accuracy plus the two balance aggregates the factory cares about: the worst-axis **floor** and the
**commit mean** (call + memory).

Method note (honest): log-prob over templated options is the validated method for the ACTION axes
(call / abstain / clarify). For open-knowledge axes (memory / lookup / unknown) templated options are
weak (the base can score ~0 even when it would answer fine) — for those a generation + LLM-judge eval
is more faithful (see the lab eval scripts). This self-contained eval is a fast in-loop check; treat
knowledge-axis absolute numbers as approximate and read base-vs-wrapper *deltas* within an axis.
"""
from __future__ import annotations

import collections
from typing import Optional

OPTION_KEYS = ("direct", "tool_call", "clarify", "refuse")
COMMIT_AXES = ("1_call", "4_memory")
DEFAULT_SYS = (
    "You are a helpful AI assistant. You may use provided tools. Answer directly only if you know; "
    "use a tool if you need to look something up; ask to clarify if ambiguous; decline if unanswerable."
)
_TYPEMAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _fix(o):
    if isinstance(o, dict):
        return {k: (_TYPEMAP.get(v, v) if k == "type" and isinstance(v, str) else _fix(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [_fix(x) for x in o]
    return o


def _to_tools(ts):
    import json
    out = []
    for t in (ts or []):
        try:
            out.append({"type": "function", "function": _fix(json.loads(t) if isinstance(t, str) else t)})
        except Exception:
            pass
    return out


def _tok_ids(tok, text, device, *, special: bool):
    """tokenizer(text) → ids on device. Tolerates fakes that don't accept add_special_tokens."""
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=special).input_ids
    except TypeError:
        ids = tok(text, return_tensors="pt").input_ids
    return ids.to(device)


def _prompt_ids(pipeline, item, system):
    tok = pipeline.tokenizer
    kw = {"tools": _to_tools(item["tools"])} if item.get("tools") else {}
    txt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": item["question"]}],
        add_generation_prompt=True, tokenize=False, **kw)
    dev = next(pipeline.model.parameters()).device
    return _tok_ids(tok, txt, dev, special=True)


def _logp(pipeline, prompt_ids, answer: str) -> float:
    import torch
    tok, model = pipeline.tokenizer, pipeline.model
    a = _tok_ids(tok, answer, prompt_ids.device, special=False)
    if a.shape[1] == 0:
        return -1e9
    full = torch.cat([prompt_ids, a], dim=1)
    out = model(input_ids=full)
    # real CausalLM → .logits; a bare-hidden model (e.g. test fake) → project via lm_head
    logits = out.logits if hasattr(out, "logits") else model.lm_head(out)
    lp = torch.log_softmax(logits[0][:-1].float(), dim=-1)   # position i predicts token i+1
    lp0 = prompt_ids.shape[1] - 1
    idx = a[0].to(lp.device)
    tok_lp = lp[lp0:lp0 + a.shape[1]][range(a.shape[1]), idx].sum()
    return float(tok_lp) / max(1, len(answer.encode("utf-8")))


def _predict(pipeline, item, *, has_doubter: bool, system: str) -> str:
    import torch
    p = _prompt_ids(pipeline, item, system)
    with torch.no_grad():
        if has_doubter:
            pipeline._run_pass1(p)
            if pipeline.collector is not None:
                pipeline.collector.freeze()
        sc = {k: _logp(pipeline, p, item["options"][k]) for k in OPTION_KEYS}
    if has_doubter and pipeline.collector is not None:
        pipeline.collector.unfreeze()
    return max(sc, key=sc.get)


def eval_suite_logprob(pipeline, suite, *, has_doubter: bool, axes: Optional[list] = None,
                       system: str = DEFAULT_SYS, verbose: bool = True) -> dict:
    """Evaluate the CURRENTLY attached pipeline state (base if detached, wrapper if attached).

    Returns {overall, by_axis, floor, commit_mean}. `axes` filters which axes to score (default: all).
    """
    items = [x for x in suite if axes is None or x["axis"] in axes]
    per = collections.defaultdict(lambda: [0, 0])
    for it in items:
        good = int(_predict(pipeline, it, has_doubter=has_doubter, system=system) == it["correct"])
        per[it["axis"]][0] += good
        per[it["axis"]][1] += 1
    by_axis = {ax: round(c[0] / c[1], 3) for ax, c in sorted(per.items())}
    overall = round(sum(c[0] for c in per.values()) / max(1, sum(c[1] for c in per.values())), 3)
    floor = round(min(by_axis.values()), 3) if by_axis else 0.0
    commit = [by_axis[a] for a in COMMIT_AXES if a in by_axis]
    commit_mean = round(sum(commit) / len(commit), 3) if commit else None
    res = {"overall": overall, "by_axis": by_axis, "floor": floor, "commit_mean": commit_mean}
    if verbose:
        print(f"  overall={overall} floor={floor} commit_mean={commit_mean} | {by_axis}", flush=True)
    return res


def compare_base_vs_doubter(pipeline, doubter, suite, *, axes: Optional[list] = None,
                            gain: float = 1.0, system: str = DEFAULT_SYS, verbose: bool = True) -> dict:
    """base (detached) vs the trained wrapper (attached @gain) on the suite. Returns {base, doubter}."""
    pipeline.detach_all()
    if verbose:
        print("  [base]", flush=True)
    base = eval_suite_logprob(pipeline, suite, has_doubter=False, axes=axes, system=system, verbose=verbose)
    pipeline.attach(doubter)
    if hasattr(doubter, "set_gain"):
        doubter.set_gain(gain)
    if verbose:
        print(f"  [doubter @{gain}]", flush=True)
    dbt = eval_suite_logprob(pipeline, suite, has_doubter=True, axes=axes, system=system, verbose=verbose)
    pipeline.detach_all()
    if verbose:
        df = dbt["floor"] - base["floor"]
        dc = (dbt["commit_mean"] or 0) - (base["commit_mean"] or 0)
        print(f"  Δ floor {df:+.3f} | Δ commit_mean {dc:+.3f}", flush=True)
    return {"base": base, "doubter": dbt}
