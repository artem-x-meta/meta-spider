"""The loop is exercised WITHOUT a model via FakePolicy — that is the whole point of the
Policy decoupling: the engine is deterministic and tested on CPU while the GPU is busy training.
"""
from daimon_agent import (
    AgentAction, BackendPolicy, FakeBackend, FakePolicy, MetaAgent,
    RefusalToolRenderer, Session, Tool, ToolRegistry, looks_like_refusal,
)


def _oracle_registry(answer="42"):
    return ToolRegistry([Tool("lookup", "oracle", lambda query: f"answer: {answer}")])


def test_refusal_triggers_tool_then_final():
    # turn 1: text refusal → renderer emits tool_call (lookup); turn 2: final.
    policy = FakePolicy(["I'm not sure about this", AgentAction.final("the answer is 42")])
    agent = MetaAgent(policy, _oracle_registry(), max_steps=4)
    res = agent.run("capital of X?")
    assert res.answer == "the answer is 42"
    assert res.steps == 2
    assert any(t["action"] == "tool" and t["tool"] == "lookup" for t in res.trace)
    # the tool observation made it into the session between turns
    assert res.trace[0]["obs"] == "answer: 42"


def test_confident_answer_is_immediate_final():
    policy = FakePolicy(["Paris is the capital"])  # not a refusal → final right away
    agent = MetaAgent(policy, _oracle_registry())
    res = agent.run("capital of France?")
    assert res.answer == "Paris is the capital"
    assert res.steps == 1
    assert res.trace[0]["action"] == "final"


def test_abstain_when_no_tool_path():
    policy = FakePolicy([AgentAction.abstain("I don't know")])
    res = MetaAgent(policy).run("?")
    assert res.abstained and res.answer == "I don't know"


def test_truncates_on_repeating_tool_calls():
    # the policy keeps calling a tool → we hit max_steps
    policy = FakePolicy([AgentAction.tool_call("lookup", {"query": "q"})] * 10)
    res = MetaAgent(policy, _oracle_registry(), max_steps=3).run("?")
    assert res.truncated and res.steps == 3


def test_backend_policy_through_inference_seam():
    # same scenario, but the decision comes from an InferenceBackend (text), not from
    # the policy script — exercises the backend → renderer → action seam.
    backend = FakeBackend(["I'm not sure", "the answer is 42"])
    agent = MetaAgent(BackendPolicy(backend), _oracle_registry(), max_steps=4)
    res = agent.run("capital of X?")
    assert res.answer == "the answer is 42"
    assert any(t["action"] == "tool" for t in res.trace)


def test_renderer_and_refusal_detector():
    assert looks_like_refusal("I'm not sure")
    assert not looks_like_refusal("The answer is clearly Paris")
    act = RefusalToolRenderer(fallback_tool="lookup").render(
        "I don't know", signal=None, session=Session().user("q"))
    assert act.kind == "tool" and act.tool == "lookup" and act.args["query"] == "q"
