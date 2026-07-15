"""Meta-Agent — agentic runtime + chat for the two-pass meta-attention wrapper.

Why a separate framework: off-the-shelf engines (vLLM/TGI, LangGraph, OpenAI-tool-loop)
rest on the contract "generate() in a single pass, decision = emitted token". The two-pass
wrapper [daimon] breaks it (hidden Pass 1, latent decision). 95% here is standard
boilerplate (loop, tools, history); the new part is a single seam — ActionRenderer (latent
decision → action). Independent of the training toolkit.

    from daimon_agent import MetaAgent, ToolRegistry, Tool, FakePolicy
    agent = MetaAgent(FakePolicy([...]), ToolRegistry([Tool("lookup", "...", fn)]))
    res = agent.run("question")
"""
__version__ = "0.0.1"

from .action import (
    AgentAction, ActionRenderer, LatentActionRenderer, ReActRenderer,
    RefusalToolRenderer, looks_like_refusal,
)
from .backends import (
    FakeBackend, InferenceBackend, InferenceOutput, StopBackend,
    LlamaCppBackend, DaimonBackend, OpenAIBackend,
)
from .native import (
    NativeToolPrompt, NativeToolRenderer, tool_schemas, STOP_SEQUENCES, stops_for,
)
from .build import assemble_agent, build_agent
from .bench import AgentBench, BenchReport, TaskOutcome
from .chat import ChatLoop
from .policy import BackendPolicy, FakePolicy, DaimonPolicy, Policy
from .prompt import GemmaFormat, PlainFormat, QwenFormat, PromptFormat, ReActPromptBuilder
from .react import build_react_agent
from .runtime import AgentResult, MetaAgent
from .serve import (ChronoAnchorControl, LatentControl, LlamaCppAnchorControl,
                    DaimonServer, build_app, from_llama_cpp, from_pipeline)
from .session import Message, Session
from .stdtools import calculator, knowledge_base
from .tasks import Task, tau_lite_suite
from .tools import Tool, ToolRegistry

__all__ = [
    "DaimonServer", "LatentControl", "ChronoAnchorControl", "LlamaCppAnchorControl",
    "from_pipeline", "from_llama_cpp", "build_app",
    "MetaAgent", "AgentResult", "build_react_agent", "assemble_agent", "build_agent",
    "Session", "Message",
    "Tool", "ToolRegistry", "calculator", "knowledge_base",
    "AgentAction", "ActionRenderer", "RefusalToolRenderer", "LatentActionRenderer",
    "ReActRenderer", "looks_like_refusal",
    "PromptFormat", "PlainFormat", "GemmaFormat", "QwenFormat", "ReActPromptBuilder",
    "NativeToolPrompt", "NativeToolRenderer", "tool_schemas", "STOP_SEQUENCES", "stops_for",
    "InferenceBackend", "InferenceOutput", "StopBackend",
    "FakeBackend", "DaimonBackend", "LlamaCppBackend", "OpenAIBackend",
    "Policy", "FakePolicy", "BackendPolicy", "DaimonPolicy",
    "Task", "tau_lite_suite", "AgentBench", "BenchReport", "TaskOutcome",
    "ChatLoop",
]
