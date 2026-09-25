"""Request validation for ``POST /v1/systemone`` (systemone-compatible request shape).

Hand-written (no pydantic) so every problem maps to the one error body the contract promises::

    HTTP 422  {"error": {"code": ..., "message": ..., "question": <id, when one question is at fault>}}

Codes: ``invalid_json`` (body is not JSON), ``invalid_request`` (top level), ``invalid_question``
(one question; ``question`` names it). Budget errors (``input_budget_exceeded``) are raised later by
the engine adapter, because they need the tokenizer.

Accepted request::

    {"model"?: str, "state": str | object | array,
     "questions": {id: {"type": "noul" | "choice" | "score",
                        "instructions": str | object | array,
                        "criteria"?: ...}}}

* noul   criteria: optional; null or an object with only ``true`` / ``false`` (each str | object | array | null).
* choice criteria: required object, 1..255 options; value = description (str | object | array) or null.
* score  criteria: required ordered array of 2..10 levels; each a non-empty string, an object
  ``{"label": str, "description"?: str}`` or any other object / array.

Unknown extra keys (top level and per question) are ignored, so clients that send additional
fields keep working. The ``model`` field is accepted and ignored (the server serves one model).

Internal mapping (what the renderer sees): ``{"t": type, "ins": str, "crit": ...}``; instructions
given as object / array become compact JSON text; choice criteria keep their order; a score level
``{"label": L, "description": D}`` becomes the text ``"L: D"`` (plain ``L`` without description),
which matches how the training data writes levels; other objects are left to the renderer's
own JSON rendering. The state is passed through unchanged: the renderer's ``serialize_state``
is the project's canonical serialisation (strings verbatim, structures as JSON).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

QTYPES = ("noul", "choice", "score")
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10


class ApiError(Exception):
    """An error with an HTTP status and the contract's JSON error body."""

    def __init__(self, status: int, code: str, message: str, question: str | None = None, **extra: Any):
        super().__init__(message)
        self.status, self.code, self.message, self.question, self.extra = status, code, message, question, extra

    def body(self) -> dict:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.question is not None:
            err["question"] = self.question
        err.update({k: v for k, v in self.extra.items() if v is not None})
        return {"error": err}


def _bad(message: str, question: str | None = None, code: str | None = None) -> ApiError:
    return ApiError(422, code or ("invalid_question" if question is not None else "invalid_request"), message,
                    question)


@dataclass
class QuestionSpec:
    id: str
    type: str
    internal: dict                      # {"t", "ins", "crit"} for the renderer / engine
    options: list[str]                  # public option keys (choice names; score level indices "0".."K-1")
    legend: dict[str, str] = field(default_factory=dict)   # score only: level index -> label


@dataclass
class ParsedRequest:
    model: str | None
    state: Any
    questions: list[QuestionSpec]


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_structured(v: Any) -> bool:
    return isinstance(v, (dict, list))


def _instructions(v: Any, qid: str) -> str:
    if isinstance(v, str):
        if not v.strip():
            raise _bad("instructions must not be empty", qid)
        return v
    if _is_structured(v):
        if not v:
            raise _bad("instructions object/array must not be empty", qid)
        return compact_json(v)
    raise _bad("instructions must be a string, object or array", qid)


def _description(v: Any, qid: str, where: str, allow_null: bool = True) -> Any:
    if v is None:
        if allow_null:
            return None
        raise _bad(f"{where} must not be null", qid)
    if isinstance(v, str) or _is_structured(v):
        return v
    raise _bad(f"{where} must be a string, object, array{' or null' if allow_null else ''}", qid)


def _level_label(level: Any) -> str:
    if isinstance(level, str):
        return level
    if isinstance(level, dict) and isinstance(level.get("label"), str):
        return level["label"]
    return compact_json(level)


def _level_text(level: Any) -> Any:
    """Renderer text for one score level (see module docstring)."""
    if isinstance(level, dict) and isinstance(level.get("label"), str) and set(level) <= {"label", "description"}:
        desc = level.get("description")
        if desc is None or desc == "":
            return level["label"]
        if not isinstance(desc, str):
            return level
        return f"{level['label']}: {desc}"
    return level


def parse_question(qid: str, q: Any) -> QuestionSpec:
    if not isinstance(q, dict):
        raise _bad("question must be an object", qid)
    t = q.get("type")
    if t not in QTYPES:
        raise _bad(f"type must be one of {', '.join(QTYPES)} (got {t!r})", qid)
    if "instructions" not in q:
        raise _bad("instructions is required", qid)
    ins = _instructions(q["instructions"], qid)
    crit = q.get("criteria")
    if t == "noul":
        if crit is None:
            return QuestionSpec(qid, t, {"t": "noul", "ins": ins, "crit": None}, ["false", "true"])
        if not isinstance(crit, dict):
            raise _bad("noul criteria must be an object with optional 'true' / 'false'", qid)
        extra = set(crit) - {"true", "false"}
        if extra:
            raise _bad(f"noul criteria accepts only 'true' and 'false' (got {sorted(extra)})", qid)
        out = {k: _description(crit.get(k), qid, f"criteria.{k}") for k in ("false", "true") if k in crit}
        return QuestionSpec(qid, t, {"t": "noul", "ins": ins, "crit": out or None}, ["false", "true"])
    if t == "choice":
        if not isinstance(crit, dict):
            raise _bad("choice criteria must be an object mapping option -> description or null", qid)
        if not 1 <= len(crit) <= MAX_CHOICE_OPTIONS:
            raise _bad(f"choice needs 1..{MAX_CHOICE_OPTIONS} options (got {len(crit)})", qid)
        out = {}
        for name, desc in crit.items():
            if not name.strip():
                raise _bad("choice option names must not be empty", qid)
            out[name] = _description(desc, qid, f"criteria[{name!r}]")
        return QuestionSpec(qid, t, {"t": "choice", "ins": ins, "crit": out}, list(out))
    # score
    if not isinstance(crit, list):
        raise _bad("score criteria must be an ordered array of levels", qid)
    if not MIN_SCORE_LEVELS <= len(crit) <= MAX_SCORE_LEVELS:
        raise _bad(f"score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels (got {len(crit)})", qid)
    for i, level in enumerate(crit):
        if isinstance(level, str):
            if not level.strip():
                raise _bad(f"score level {i} must not be empty", qid)
        elif isinstance(level, dict) and "label" in level:
            if not isinstance(level["label"], str) or not level["label"].strip():
                raise _bad(f"score level {i}: label must be a non-empty string", qid)
            if level.get("description") is not None and not isinstance(level["description"], str):
                raise _bad(f"score level {i}: description must be a string", qid)
        elif not (_is_structured(level) and level):
            raise _bad(f"score level {i} must be a non-empty string, {{label, description}} object or array", qid)
    names = [str(i) for i in range(len(crit))]
    legend = {n: _level_label(level) for n, level in zip(names, crit)}
    return QuestionSpec(qid, t, {"t": "score", "ins": ins, "crit": [_level_text(x) for x in crit]}, names, legend)


def parse_request(body: Any) -> ParsedRequest:
    if not isinstance(body, dict):
        raise _bad("request body must be a JSON object")
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise _bad("model must be a string")
    if "state" not in body:
        raise _bad("state is required")
    state = body["state"]
    if not (isinstance(state, str) or _is_structured(state)):
        raise _bad("state must be a string, object or array")
    qs = body.get("questions")
    if not isinstance(qs, dict):
        raise _bad("questions must be an object mapping id -> question")
    if not qs:
        raise _bad("questions must not be empty")
    specs = []
    for qid, q in qs.items():
        if not qid:
            raise _bad("question ids must be non-empty strings")
        specs.append(parse_question(qid, q))
    return ParsedRequest(model=model, state=state, questions=specs)


def parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise ApiError(422, "invalid_json", f"request body is not valid JSON: {e}") from None
