"""Контракт границы: meta-attention (МЕХАНИЗМ) ↔ daimon (ГОЛОСА).

Механизм вынесен в отдельную библиотеку (Apache-2.0) именно потому, что он не зависит от того,
ЧТО в него инъектируют. Эти тесты стерегут границу с обеих сторон:

    библиотека не знает о голосах       ← иначе разрез проведён неверно
    голоса удовлетворяют её протоколу   ← иначе они не подключатся
"""
import pytest


# ───────────────────────── библиотека не знает о голосах ─────────────────────────

def test_the_library_never_imports_the_framework():
    """meta_attention не тянет daimon — ни прямо, ни транзитивно."""
    import subprocess
    import sys

    code = ("import meta_attention, sys; "
            "leaked = [m for m in sys.modules if m.startswith('daimon')]; "
            "print(leaked)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"библиотека притащила фреймворк: {out.stdout}"


def test_the_mechanism_knows_nothing_about_doubt_goals_or_memory():
    import meta_attention

    exported = " ".join(meta_attention.__all__).lower()
    for concept in ("doubter", "goal", "chrono", "memory", "voice", "daimon"):
        assert concept not in exported, f"механизм не должен экспортировать «{concept}»"


# ───────────────────────── голоса стоят на протоколе механизма ─────────────────────────

def test_voice_satisfies_the_injector_protocol():
    """Voice — это Injector плюс мнение фреймворка (жизненный цикл, чекпоинт, гейн)."""
    from meta_attention import Injector
    from daimon_voices import Voice

    for hook in ("on_attach", "on_detach", "on_pre_forward", "on_post_forward"):
        assert hasattr(Voice, hook), f"Voice обязан реализовать {hook} из протокола"
    assert hasattr(Voice, "set_gain"), "гейн-фейдер — добавка фреймворка, не механизма"
    assert Injector is not Voice                      # интерфейс ≠ его расширение


@pytest.mark.parametrize("voice_name", ["Doubter", "GoalAnchor", "Chronographer", "ChronoAnchor"])
def test_every_voice_is_attachable(voice_name, fake_lm_factory):
    """Каждый голос структурно подходит механизму (иначе pipeline.attach его не примет)."""
    import daimon_voices
    from meta_attention import Injector

    cls = getattr(daimon_voices, voice_name)
    for hook in ("on_attach", "on_detach", "on_post_forward"):
        assert callable(getattr(cls, hook, None)), f"{voice_name}: нет {hook}"


# ───────────────────────── публичный API фреймворка ─────────────────────────

def test_umbrella_reexports_both_sides_as_the_same_objects():
    """`daimon` — единая точка входа: механизм под своими именами + голоса."""
    import daimon
    import daimon_voices
    import meta_attention

    assert daimon.DaimonPipeline is meta_attention.MetaAttentionPipeline
    assert daimon.DaimonConfig is meta_attention.MetaAttentionConfig
    assert daimon.Doubter is daimon_voices.Doubter
    assert daimon.ChronoAnchor is daimon_voices.ChronoAnchor


def test_canonical_voice_configs():
    from daimon_voices import ChronoAnchorConfig, DoubterConfig, GoalAnchorConfig

    assert DoubterConfig().encoder_type == "selective"
    assert GoalAnchorConfig().trigger == "fixed"
    assert ChronoAnchorConfig().goal_pool == "last"


def test_watchdog_is_not_a_voice():
    """Пробы не говорят — они ГЕЙТЯТ. Поэтому watchdog живёт в голосах, но голосом не является."""
    from daimon_voices.watchdog import ConfidenceProbe
    from daimon_voices import Voice

    assert not issubclass(ConfidenceProbe, Voice)
    assert not hasattr(ConfidenceProbe, "on_post_forward")   # он не инъектирует
