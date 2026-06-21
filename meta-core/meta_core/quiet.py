"""Suppress noisy/scary transformers logs in the CLI (friction F1/F5).

F1: dozens of "Setting pad_token_id to eos_token_id" on per-call generate → looks like a hang.
F5: transformers spews "[ERROR] … is part of …'s signature, but not documented" (auto_docstring
    validator) — these are NOT errors, but a random dev panics; + "fast path is not available" (no
    flash-linear-attention). All harmless.

`quiet_transformers()` is called at the start of the metaloom/meta-agent CLI. It does not touch our print logs.
"""
from __future__ import annotations

import logging


class _DropBenignDocstringErrors(logging.Filter):
    """Filters out the harmless auto_docstring "errors" (scary, but not errors)."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        m = record.getMessage()
        drop = ("but not documented" in m) or ("is part of" in m and "signature" in m)
        return not drop


def quiet_transformers() -> None:
    """Quiet down transformers noise: verbosity→error (hides pad_token/fast-path) + filter for
    harmless docstring "[ERROR]"s. Idempotent, safe when transformers is absent."""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    flt = _DropBenignDocstringErrors()
    for name in ("", "transformers", "transformers.modeling_utils", "transformers.utils"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _DropBenignDocstringErrors) for f in lg.filters):
            lg.addFilter(flt)
