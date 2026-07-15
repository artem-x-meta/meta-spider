"""Ready deterministic tools for agentic tasks/demos/tests.

Real (not oracle) executor tools: Calculator evaluates arithmetic, KnowledgeBase looks
up a fact by key. Pure plumbing — no connection to the model.
"""
from __future__ import annotations

import ast
import operator

from .tools import Tool

__all__ = ["calculator", "knowledge_base"]

# --- safe arithmetic (no eval()) ---
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _ev(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    raise ValueError("unsupported expression")


def _calc(expression: str) -> str:
    try:
        val = _ev(ast.parse(str(expression), mode="eval").body)
    except Exception as e:
        return f"[calc error: {e}]"
    # integers without a trailing .0
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val)


def calculator() -> Tool:
    """Calculator tool. Argument: expression (e.g. '23*17+5')."""
    return Tool("calculator",
                "Evaluate an arithmetic expression. Argument: expression (e.g. '23*17+5').",
                _calc, arg="expression")


def knowledge_base(facts: dict, name: str = "lookup") -> Tool:
    """Fact-lookup-by-key tool. Argument: key. Keys are normalized (lower/strip)."""
    norm = {str(k).strip().lower(): str(v) for k, v in facts.items()}

    def _lookup(key: str) -> str:
        return norm.get(str(key).strip().lower(), f"[not found: {key}]")

    return Tool(name, "Look up a fact by key in the knowledge base. Argument: key.", _lookup, arg="key")
