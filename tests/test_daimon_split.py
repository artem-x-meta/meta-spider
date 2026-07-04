"""Контракт разреза meta-core (механизм) / meta-daimon (голоса)."""
import pytest


def test_canonical_imports():
    from meta_daimon import Doubter, DoubterConfig
    assert DoubterConfig().encoder_type == "selective"


def test_backcompat_shims_same_objects():
    """Старые пути (meta_core, meta_core.modifiers, meta_spider) ведут В ТОТ ЖЕ класс."""
    from meta_daimon import Doubter as canonical
    from meta_core import Doubter as via_core
    from meta_core.modifiers import Doubter as via_modifiers
    from meta_spider import Doubter as via_umbrella
    assert canonical is via_core is via_modifiers is via_umbrella


def test_core_is_mechanism_only():
    """В meta-core не осталось модулей голосов; контракт Modifier — остался."""
    import meta_core.modifiers as mods
    from meta_core.modifiers.base import Modifier          # noqa: F401
    with pytest.raises(ImportError):
        import meta_core.modifiers.doubter                  # noqa: F401
    with pytest.raises(AttributeError):
        mods.NoSuchVoice