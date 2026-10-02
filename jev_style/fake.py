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

    def __init__(self, salt: str = "") -> None:
        self.renderer = _Renderer()
        self.salt = salt            # a different salt = a different (still deterministic) "model"

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
                raw.append(_score(self.salt + s, q.get("ins"), n))
            probs = _softmax(raw)
            out.append({"answer": names[probs.index(max(probs))], "probabilities": dict(zip(names, probs)),
                        "input_tokens": n_state + head, "head_tokens": head})
        return out


def _score(state: str, ins: Any, option: str) -> float:
    h = hashlib.sha256(f"{state}\x00{ins}\x00{option}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 * 4.0


def _softmax(raw: list[float]) -> list[float]:
    m = max(raw)
    e = [math.exp(x - m) for x in raw]
    z = sum(e)
    return [v / z for v in e]


def jevk5_response(body: Any, salt: str = "jevk5", model: str = "alibiserikbay/JevK5-9B") -> dict:
    """What ``jevk5-serve`` (allebee/jevk5 0.3.3) returns for ``body``, with hash-based probabilities. Same shapes:
    every answer has ``confidence`` = p_max; noul carries only ``noul``; choice ``choice`` + ``probabilities``;
    score ``score`` + ``probabilities`` keyed "0".."K-1" (no ``legend``); ``usage`` has no ``state_tokens``.
    Invalid input raises ValueError (the real server answers 400 ``{"error": "<message>"}``)."""
    if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
        raise ValueError("'state'" if isinstance(body, dict) and "state" not in body else "'questions'")
    s = body["state"] if isinstance(body["state"], str) else json.dumps(body["state"], ensure_ascii=False)
    answers, tokens = {}, 0
    for qid, q in body["questions"].items():
        t, crit = q.get("type"), q.get("criteria")
        if t not in ("noul", "choice", "score"):
            raise ValueError(f"unknown question type {t!r}")
        if "instructions" not in q:
            raise ValueError("question is missing instructions")
        if t == "choice":
            crit = dict.fromkeys(crit) if isinstance(crit, list) else crit
            if not isinstance(crit, dict) or len(crit) < 2:
                raise ValueError("choice criteria must name at least two options")
        if t == "score" and (not isinstance(crit, list) or len(crit) < 2):
            raise ValueError("score criteria must list at least two levels")
        names = (["true", "false"] if t == "noul" else list(crit) if t == "choice"
                 else [str(i) for i in range(len(crit))])
        probs = dict(zip(names, _softmax([_score(salt + s, json.dumps(q.get("instructions")), n) for n in names])))
        a: dict[str, Any] = {"type": t, "confidence": max(probs.values())}
        if t == "noul":
            a["noul"] = probs["true"]
        elif t == "choice":
            a.update(choice=max(probs, key=probs.get), probabilities=probs)
        else:
            a.update(score=sum(int(k) * v for k, v in probs.items()), probabilities=probs)
        answers[qid] = a
        tokens += _tokens(s) + 8 * len(names)
    return {"model": body.get("model") or model, "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0}, "latency_ms": 0.1}
