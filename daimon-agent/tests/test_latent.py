"""LatentActionRenderer: decision from the latent signal, priority over text, fall back
to text when there is no signal. Tested via FakeBackend emitting (text, signal)."""
from daimon_agent import (
    AgentAction, BackendPolicy, FakeBackend, LatentActionRenderer, MetaAgent,
    Session, Tool, ToolRegistry,
)


def _oracle():
    return ToolRegistry([Tool("lookup", "oracle", lambda query: "answer: 42")])


def _render(text, signal, threshold=0.5):
    r = LatentActionRenderer(fallback_tool="lookup", threshold=threshold)
    return r.render(text, signal, Session().user("question"))


def test_latent_confidence_low_refuses():
    act = _render("Paris", {"confidence": 0.2})       # confidence < 0.5 → refusal
    assert act.kind == "tool" and act.tool == "lookup" and act.args["query"] == "question"


def test_latent_confidence_high_answers():
    act = _render("Paris", {"confidence": 0.9})       # confident → final (text ignored)
    assert act.kind == "final" and act.content == "Paris"


def test_latent_explicit_refuse_flag():
    assert _render("anything", {"refuse": True}).kind == "tool"
    assert _render("anything", {"refuse": False}).kind == "final"


def test_latent_signal_beats_text():
    # the text sounds like a refusal, but the latent signal says "confident" → latent wins
    act = _render("I'm not sure", {"confidence": 0.95})
    assert act.kind == "final" and act.content == "I'm not sure"


def test_no_signal_falls_back_to_text():
    # no signal → behave like RefusalToolRenderer (by text)
    assert _render("I don't know", None).kind == "tool"        # text refusal → tool
    assert _render("Paris is the capital", None).kind == "final"


def test_latent_through_backend_policy():
    # FakeBackend returns (text, signal); the loop handles latent refusal→tool→final
    backend = FakeBackend([("draft", {"confidence": 0.1}), ("the answer is 42", {"confidence": 0.9})])
    agent = MetaAgent(BackendPolicy(backend, LatentActionRenderer()), _oracle(), max_steps=4)
    res = agent.run("capital of X?")
    assert res.answer == "the answer is 42"
    assert any(t["action"] == "tool" for t in res.trace)
