"""ReAct layer: Action/Answer parsing, prompt assembly, end-to-end multi-tool
run. A backend speaking the ReAct format = a "real model" in miniature (no GPU)."""
from daimon_agent import (
    AgentAction, FakeBackend, GemmaFormat, ReActPromptBuilder, ReActRenderer,
    Session, Tool, ToolRegistry, build_react_agent, calculator, knowledge_base,
)


def _tools():
    return ToolRegistry([calculator(),
                         knowledge_base({"france_capital_population": "2100000"})])


# --- model output parsing ---
def test_react_parses_action_to_right_arg():
    r = ReActRenderer(_tools())
    act = r.render("Action: calculator[23*17+5]", None, Session().user("q"))
    assert act.kind == "tool" and act.tool == "calculator"
    assert act.args == {"expression": "23*17+5"}        # arg placed under tool.arg name


def test_react_parses_lookup_and_answer():
    r = ReActRenderer(_tools())
    a1 = r.render("Action: lookup[speed_of_light]", None, Session())
    assert a1.tool == "lookup" and a1.args == {"key": "speed_of_light"}
    a2 = r.render("Answer: 42", None, Session())
    assert a2.kind == "final" and a2.content == "42"


def test_react_plain_text_is_final_and_refusal_abstains():
    r = ReActRenderer(_tools())
    assert r.render("Paris.", None, Session()).kind == "final"
    assert r.render("I'm not sure", None, Session()).kind == "abstain"


# --- prompt assembly ---
def test_prompt_lists_tools_and_format():
    pb = ReActPromptBuilder(_tools(), fmt=GemmaFormat())
    session = Session().user("What is 2+2?")
    p = pb(session)
    assert "calculator" in p and "lookup" in p            # tool schema
    assert "Action:" in p and "Answer:" in p              # format instructions
    assert "User: What is 2+2?" in p                      # transcript
    assert p.startswith("<start_of_turn>user")            # Gemma wrapper


def test_prompt_renders_action_observation_transcript():
    pb = ReActPromptBuilder(_tools())
    s = Session().user("q")
    s.add("assistant", "lookup[speed_of_light]", name="action")
    s.tool("lookup", "299792458")
    p = pb(s)
    assert "Action: lookup[speed_of_light]" in p
    assert "Observation: 299792458" in p


# --- end-to-end run: a "model" in ReAct format solves a chain ---
def test_react_agent_solves_chain_end_to_end():
    backend = FakeBackend([
        "Action: lookup[france_capital_population]",
        "Action: calculator[2100000*3]",
        "Answer: 6300000",
    ])
    agent = build_react_agent(backend, _tools(), fmt=GemmaFormat(), max_steps=6)
    res = agent.run("Take france_capital_population from the base and multiply by 3.")
    assert res.answer == "6300000"
    assert sum(1 for t in res.trace if t["action"] == "tool") == 2
    # the tools actually executed: lookup → 2100000, calc → 6300000
    assert res.trace[0]["obs"] == "2100000" and res.trace[1]["obs"] == "6300000"
