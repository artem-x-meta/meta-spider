"""tau-bench-lite: real tools + multi-turn tasks + scoring. Without GPU."""
from daimon_agent import (
    AgentAction, AgentBench, FakePolicy, MetaAgent, Task, ToolRegistry,
    calculator, knowledge_base, tau_lite_suite,
)


def test_calculator_tool():
    calc = calculator()
    assert calc(expression="23*17+5") == "396"
    assert calc(expression="(6371)*2") == "12742"
    assert calc(expression="10/4") == "2.5"
    assert "error" in calc(expression="import os").lower()  # not eval(), safe


def test_knowledge_base_tool():
    kb = knowledge_base({"Speed_Of_Light": "299792458"})
    assert kb(key="speed_of_light") == "299792458"   # key normalization
    assert "not found" in kb(key="unknown")


def test_task_success_check():
    t = Task("x", "...", "396")
    assert t.succeeded("the answer is 396")
    assert not t.succeeded("397")


def test_suite_loads():
    tasks, facts = tau_lite_suite()
    assert len(tasks) >= 3
    assert "speed_of_light" in facts


def test_bench_scores_single_calc_task():
    # FakePolicy scripts tool_call→final; the bench counts success + tool_calls
    tools = ToolRegistry([calculator()])
    policy = FakePolicy([
        AgentAction.tool_call("calculator", {"expression": "23*17+5"}),
        AgentAction.final("396"),
    ])
    agent = MetaAgent(policy, tools)
    report = AgentBench([Task("calc", "Compute 23*17+5.", "396", tools=["calculator"])]).run(agent)
    assert report.n == 1
    assert report.success_rate == 1.0
    assert report.total_tool_calls == 1
    assert report.outcomes[0].success and report.outcomes[0].answer == "396"


def test_bench_marks_failure():
    tools = ToolRegistry([calculator()])
    policy = FakePolicy([AgentAction.final("wrong")])
    report = AgentBench([Task("calc", "Compute 2+2.", "4")]).run(MetaAgent(policy, tools))
    assert report.success_rate == 0.0
    assert not report.outcomes[0].success
