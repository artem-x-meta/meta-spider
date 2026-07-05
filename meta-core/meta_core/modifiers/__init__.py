"""The Modifier CONTRACT — the mechanism-side plug for Meta-Daimon voices.

`Modifier` is the abstract lifecycle every voice implements (on_attach / on_pre_forward /
on_post_forward / on_detach + step hooks); `MetaSpiderPipeline.attach()` speaks only this
contract. The concrete voices (Doubter, GoalAnchor, …) live in the **meta-daimon** package —
see its docstring for the leg's philosophy (Socratic daimonion: counsels, doesn't rule).

Back-compat: `from meta_core.modifiers import Doubter / GoalAnchor` is forwarded lazily
to `meta_daimon` when it is installed.
"""

from meta_core.modifiers.base import Modifier

_DAIMON_NAMES = ("Doubter", "GoalAnchor")

__all__ = ["Modifier", *_DAIMON_NAMES]


def __getattr__(name: str):
    if name in _DAIMON_NAMES:
        try:
            import meta_daimon
        except ImportError as e:
            raise ImportError(
                f"meta_core.modifiers.{name} moved to the meta-daimon package; "
                f"import it from `meta_daimon`.") from e
        return getattr(meta_daimon, name)
    raise AttributeError(f"module 'meta_core.modifiers' has no attribute {name!r}")
