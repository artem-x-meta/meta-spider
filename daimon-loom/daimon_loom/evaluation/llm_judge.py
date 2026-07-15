"""OpenRouterJudge — LLM-as-judge via OpenRouter (free tier).

For subjective checks (persona / tone / language / approach divergence) there are no programmatic
rules — we use a 120B-class judge via OpenRouter:

  - nvidia/nemotron-3-super-120b-a12b:free  (default)
  - openai/gpt-oss-120b:free                (backup)
  - z-ai/glm-4.5-air:free                   (alternative)

Architecture:
  - HTTP client to openrouter.ai/api/v1/chat/completions (OpenAI-compatible)
  - Sequential calls with throttle (free tier rate-limit ~20 req/min = 3s between requests)
  - Output parsing: the last number 0-10 in the response → float [0, 1]
  - Retry on transient errors (429, 5xx) with exponential backoff
  - In-memory cache (text+criterion → score) for repeated requests

The API key is read from:
  1. The constructor's `api_key` parameter (priority)
  2. Env var `OPENROUTER_API_KEY`
  3. A `.env` file in the current directory or its parents

Usage:
    judge = OpenRouterJudge()
    score = judge.judge("Arrr matey!", criterion="Speak like a pirate")
    # → 1.0
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Literal, Optional

__all__ = ["OpenRouterJudge"]


JudgeModel = Literal[
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "z-ai/glm-4.5-air:free",
]


_JUDGE_SYS = (
    "You are an evaluator. Given a rule and a text, decide whether the text follows the rule. "
    "Respond with a single integer 0-10 (0 = violates completely, 10 = follows perfectly). "
    "Output ONLY the number, no explanation, no other text."
)

_JUDGE_PAIR_SYS = (
    "You are an evaluator. Given a comparison criterion and two texts (A and B), "
    "score how strongly criterion applies to B relative to A on a 0-10 scale "
    "(0 = no difference, 10 = maximally different). "
    "Output ONLY the number, no explanation."
)

_NUM_PAT = re.compile(r"\d+")


def _find_env_file() -> Optional[Path]:
    """Search for a .env file starting from CWD and going up to the root."""
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        candidate = p / ".env"
        if candidate.exists():
            return candidate
    return None


def _resolve_api_key(api_key: Optional[str]) -> str:
    """Get the API key from the parameter / env / .env file."""
    if api_key:
        return api_key
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    env_path = _find_env_file()
    if env_path is not None:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Set the env var or "
        "pass api_key to the constructor."
    )


class OpenRouterJudge:
    """LLM-as-judge via OpenRouter free models.

    Args:
        model: model name (default: Nemotron 120B Super free).
        api_key: if None — taken from env / .env (see `_resolve_api_key`).
        request_interval_sec: throttle between requests (default 0.5s — for the free tier).
        max_retries: retries on 429/5xx (default 4 with exp backoff).
    """

    DEFAULT_MODEL: JudgeModel = "nvidia/nemotron-3-super-120b-a12b:free"
    BACKUP_MODEL: JudgeModel = "openai/gpt-oss-120b:free"
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: JudgeModel = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        request_interval_sec: float = 0.5,
        max_retries: int = 4,
    ):
        self.model = model
        self.api_key = _resolve_api_key(api_key)
        self.request_interval = request_interval_sec
        self.max_retries = max_retries
        self._last_call = 0.0
        self._cache: dict[tuple[str, str], float] = {}
        self._calls_total = 0
        self._calls_failed = 0

    # ============================================================
    # Public API
    # ============================================================

    def judge(self, text: str, criterion: str) -> float:
        """Score a single text against the criterion. Returns score ∈ [0, 1]."""
        cache_key = (text, criterion)
        if cache_key in self._cache:
            return self._cache[cache_key]
        messages = self._build_single_messages(criterion, text)
        score = self._call_with_retry(messages)
        self._cache[cache_key] = score
        return score

    def judge_pair(self, text_a: str, text_b: str, criterion: str) -> float:
        """Compare two generations against the criterion (e.g. 'different reasoning approach').

        Returns 0..1 — how much B differs from A on the criterion.
        """
        cache_key = (f"PAIR:{text_a}|||{text_b}", criterion)
        if cache_key in self._cache:
            return self._cache[cache_key]
        messages = self._build_pair_messages(criterion, text_a, text_b)
        score = self._call_with_retry(messages)
        self._cache[cache_key] = score
        return score

    def judge_batch(self, texts: list[str], criterion: str) -> list[float]:
        """Sequential batch — sequential because of the free-tier rate limit."""
        return [self.judge(t, criterion) for t in texts]

    def stats(self) -> dict:
        return {
            "calls_total": self._calls_total,
            "calls_failed": self._calls_failed,
            "cache_hits": len(self._cache),
            "model": self.model,
        }

    # ============================================================
    # Internal: messages builder + HTTP + parsing
    # ============================================================

    def _build_single_messages(self, criterion: str, text: str) -> list[dict]:
        return [
            {"role": "system", "content": _JUDGE_SYS},
            {"role": "user", "content": f"Rule: {criterion}\n\nText:\n{text}\n\nScore (0-10):"},
        ]

    def _build_pair_messages(self, criterion: str, a: str, b: str) -> list[dict]:
        return [
            {"role": "system", "content": _JUDGE_PAIR_SYS},
            {"role": "user", "content": (
                f"Criterion: {criterion}\n\n"
                f"TEXT A:\n{a}\n\n"
                f"TEXT B:\n{b}\n\n"
                f"Score (0-10):"
            )},
        ]

    def _parse_score(self, response: str) -> float:
        """Parse a number 0-10 from the response (take the last valid one)."""
        matches = _NUM_PAT.findall(response.strip())
        if not matches:
            return 0.5
        valid = [int(m) for m in matches if 0 <= int(m) <= 10]
        if not valid:
            return 0.5
        return valid[-1] / 10.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_call = time.time()

    def _call_api(self, messages: list[dict]) -> str:
        import urllib.error
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.ENDPOINT, data=payload, method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/local/daimon",
                "X-Title": "daimon-judge",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def _call_with_retry(self, messages: list[dict]) -> float:
        import urllib.error

        for attempt in range(self.max_retries):
            try:
                self._throttle()
                self._calls_total += 1
                response = self._call_api(messages)
                return self._parse_score(response)
            except urllib.error.HTTPError as e:
                if e.code in (429, 502, 503, 504) and attempt < self.max_retries - 1:
                    wait = (2 ** attempt) * 5
                    time.sleep(wait)
                    continue
                self._calls_failed += 1
                return 0.5
            except Exception:
                self._calls_failed += 1
                return 0.5
        return 0.5
