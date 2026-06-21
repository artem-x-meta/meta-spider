"""Chat = the same agentic loop; the "tool" that supplies the next message is the
human. So there is no separate engine, just a thin wrapper over MetaAgent with a shared
session across turns.
"""
from __future__ import annotations

from typing import Optional

from .runtime import AgentResult, MetaAgent
from .session import Session

__all__ = ["ChatLoop"]


class ChatLoop:
    def __init__(self, agent: MetaAgent, session: Optional[Session] = None):
        self.agent = agent
        self.session = session or Session()

    def send(self, user_input: str) -> AgentResult:
        """One turn: the shared session keeps history across messages."""
        return self.agent.run(user_input, session=self.session)

    def repl(self) -> None:  # pragma: no cover - interactive
        print("Meta-Agent chat. Ctrl-C to exit.")
        try:
            while True:
                user_input = input("you> ").strip()
                if not user_input:
                    continue
                res = self.send(user_input)
                tag = "[abstain] " if res.abstained else ""
                print(f"bot> {tag}{res.answer}")
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
