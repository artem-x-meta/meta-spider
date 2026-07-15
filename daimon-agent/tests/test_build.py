"""daimon-agent practical layer: assemble_agent (builder) + CLI parser. Without GPU (FakeBackend)."""
from daimon_agent import FakeBackend, Tool, ToolRegistry, assemble_agent
from daimon_agent.cli import build_parser


class FakeTok:
    """Minimal tokenizer: apply_chat_template ignores content (FakeBackend is scripted anyway)."""
    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=True, **kw):
        return "PROMPT"


def test_assemble_agent_chat():
    """Clean chat (no tools): backend returns text → final answer."""
    agent = assemble_agent(FakeBackend(["Hi! How can I help?"]), FakeTok(), model_name="qwen")
    res = agent.run("hello")
    assert res.answer == "Hi! How can I help?", res.answer


def test_assemble_agent_tool_loop():
    """With a tool: native tool_call → execution → final."""
    be = FakeBackend([
        "<function=calc>\n<parameter=expr>\n2+2\n</parameter>\n</function>",
        "the answer is 4",
    ])
    reg = ToolRegistry([Tool("calc", "calculate", lambda expr: "4", arg="expr")])
    agent = assemble_agent(be, FakeTok(), tools=reg, model_name="qwen", max_steps=3)
    res = agent.run("2+2?")
    assert "4" in res.answer, res.answer


def test_cli_parser_chat_and_run():
    ap = build_parser()
    a = ap.parse_args(["chat", "--model", "Qwen/Qwen3.5-4B", "--checkpoint", "d.pt",
                       "--quantization", "nf4"])
    assert a.cmd == "chat" and a.model == "Qwen/Qwen3.5-4B" and a.checkpoint == "d.pt"
    assert a.quantization == "nf4"
    b = ap.parse_args(["run", "--model", "Y", "solve this"])
    assert b.cmd == "run" and b.task == "solve this" and b.checkpoint is None
