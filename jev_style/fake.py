"""A deterministic stand-in for the model: same interface and budgets, no weights, no torch.

Used by ``jev-style serve --fake``, the tests and CI. Probabilities come from a hash of
(state, question, option), so answers are stable but meaningless. Never use it to judge quality.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

FAKE_MODEL_ID = "jev-style-fake"
CONTEXT_LIMIT = 25_600
HARD_HEAD_MAX = 2_048


class InputBudgetError(ValueError):
    pass


class QuestionError(ValueError):
    pass


def _tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))          # rough chars-per-token estimate


def _names(q: dict) -> list[str]:
    t, crit = q.get("t"), q.get("crit")
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise QuestionError("choice needs options")
        return [str(k) for k in crit]
    if t == "score":
        if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
            raise QuestionError("score needs 2..10 levels")
        return [str(i) for i in range(len(crit))]
    if t == "noul":
        return ["false", "true"]
    raise QuestionError(f"unknown question type {t!r}")


class _Renderer:
    max_len = CONTEXT_LIMIT
    head_max = HARD_HEAD_MAX


class FakeRuntime:
    backend = "fake"

    def __init__(self) -> None:
        self.renderer = _Renderer()

    def decide_many(self, state: Any, questions: list[dict], category: str | None = None) -> list[dict]:
        s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        n_state = _tokens(s)
        out = []
        for q in questions:
            names = _names(q)
            head = _tokens(json.dumps(q, ensure_ascii=False)) + 4 * len(names)
            if head > HARD_HEAD_MAX:
                raise InputBudgetError(f"question/options need {head} tokens; head budget={HARD_HEAD_MAX}")
            if n_state + head > CONTEXT_LIMIT:
                raise InputBudgetError(f"input needs {n_state + head} tokens; max_len={CONTEXT_LIMIT}")
            raw = []
            for n in names:
                h = hashlib.sha256(f"{s}\x00{q.get('ins')}\x00{n}".encode()).digest()
                raw.append(int.from_bytes(h[:4], "big") / 2**32 * 4.0)
            m = max(raw)
            e = [math.exp(x - m) for x in raw]
            z = sum(e)
            out.append({"answer": names[e.index(max(e))], "probabilities": {n: v / z for n, v in zip(names, e)},
                        "input_tokens": n_state + head, "head_tokens": head})
        return out
