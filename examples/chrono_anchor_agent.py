"""ChronoAnchor in an agentic session — the reference wiring.

The goal lives ONLY in the latent channel: it never enters the prompt, the system message or
any tool result. The agent still obeys it, because every memory token the organ writes carries
a piece of the goal (goal-conditioned episode compression).

    python examples/chrono_anchor_agent.py --checkpoint chrono_anchor.pt \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --goal "Stay under $60 in total; never pick the allergenic color."

Wiring, in three lines:
    anchor = ChronoAnchor.from_checkpoint(ckpt);  pipe.attach(anchor)
    anchor.set_goal(goal)                                  # once per session
    MetaAgent(policy, tools, step_hooks=[anchor]).run(task) # one episode per step
"""
from __future__ import annotations

import argparse

from daimon_agent import MetaAgent
from daimon_agent.backends import DaimonBackend
from daimon_agent.native import NativeToolPrompt, NativeToolRenderer
from daimon_agent.policy import BackendPolicy
from daimon_agent.tools import Tool, ToolRegistry
from meta_attention import MetaAttentionConfig, MetaAttentionPipeline
from daimon_voices import ChronoAnchor

CATALOG = {
    "desk lamp": [
        {"id": 1, "name": "blue edition", "price": 30},
        {"id": 2, "name": "PREMIUM white edition", "price": 150},
        {"id": 3, "name": "standard white edition", "price": 45},
    ],
}


def search_catalog(query: str) -> str:
    items = CATALOG.get(query.strip().lower(), [])
    if not items:
        return f"no listings for {query!r}"
    return "; ".join(f"[{i['id']}] {i['name']} — ${i['price']}" for i in items)


def profile_lookup(user: str) -> str:
    # The FACT lives in the transcript (episodic memory carries it through the session);
    # the GOAL lives in the latent channel (never in text).
    return "profile: allergic to the blue coating — never pick blue items"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--checkpoint", required=True, help="trained ChronoAnchor .pt")
    ap.add_argument("--goal", required=True, help="the session goal — NEVER shown to the model")
    ap.add_argument("--task", default="Buy me a desk lamp.")
    ap.add_argument("--quantization", default=None, choices=[None, "nf4", "int8"])
    args = ap.parse_args()

    cfg = MetaAttentionConfig(model_name=args.model, device="cuda", dtype="float16",
                           target_layers="late", cross_attn_layers="late",
                           quantization=args.quantization)
    pipe = MetaAttentionPipeline.from_pretrained(cfg)

    anchor = ChronoAnchor.from_checkpoint(args.checkpoint)
    pipe.attach(anchor)
    anchor.set_goal(args.goal)          # ← the whole "goal in the session" API

    tools = ToolRegistry([
        Tool(name="search_catalog", description="List the catalog entries for a product.",
             fn=search_catalog, arg="query"),
        Tool(name="profile_lookup", description="Look up the buyer's profile notes.",
             fn=profile_lookup, arg="user"),
    ])
    prompt = NativeToolPrompt(pipe.tokenizer, tools=list(tools),
                              system="You are a shopping agent. Use the tools, then state the "
                                     "single option you buy.")
    policy = BackendPolicy(DaimonBackend(pipe, max_new_tokens=96),
                           renderer=NativeToolRenderer(tools=list(tools)),
                           prompt_builder=prompt)

    agent = MetaAgent(policy, tools, max_steps=6, step_hooks=[anchor])   # ← episodes per step
    result = agent.run(args.task)

    print("\n=== session ===")
    for t in result.trace:
        print(f"  step {t['step']}: {t['action']}"
              + (f" {t.get('tool')}({t.get('args')}) → {t.get('obs')}" if t["action"] == "tool" else ""))
    print(f"\nGOAL (latent only, never in the prompt): {args.goal}")
    print(f"ANSWER: {result.answer}")
    print(f"bank: {len(anchor.bank)} episodes, memory {anchor._current_memory.size(0)} tokens")


if __name__ == "__main__":
    main()
