"""Confidence cascade: each question is answered by the smallest tier that is confident enough.

A cascade is an ordered list of tiers (small -> large), each a systemone engine: an in-process release
(``local:2b-v3``), a Jev-Style repo (``hf:<repo>``), JevK5 in-process (``jevk5:<repo or folder>``, see
``jev_style.jevk5_engine``) or any server that implements ``POST /v1/systemone`` (``http://host:port``:
``jev-style serve``, ``jevk5-serve``, ...). One request::

    tier 1 answers every question in ONE call (the state is read once)
    the questions it was unsure of (or failed on) go to tier 2 in ONE call: same state, only those questions
    ... and so on up to the top tier

Routing rules (pre-declared; ``select`` / ``keep`` are the only place they live, so the live cascade and an offline
replay of recorded per-question answers route the same way):

* Confidence is computed here from the tier's probabilities, never read from its ``confidence`` field:
  choice / score with k options ``(k * p_max - 1) / (k - 1)``; noul ``|2 * P(true) - 1|`` (the same at k = 2).
  Probabilities are renormalised once (as ``Adapter`` does); a malformed answer counts as a tier error.
* One threshold per non-top tier, shared by every question type: tier i keeps an answer iff
  confidence >= thresholds[i]; otherwise the question escalates.
* A tier error for a question (input budget, invalid or unsupported question, the HTTP error of the request it
  was in, a malformed answer) escalates it as well. A request-level error fails every question of that request.
* The top tier's answer is final. If the top tier fails, the answer of the last tier that succeeded is used.
  Only when every tier failed is an error returned, as ``Adapter`` returns it today: the error of the lowest tier
  that ran the question (what a single-model server would have said), plus ``tier``, ``tier_model`` and
  ``tier_errors``.
* Optional per-tier ``max_options`` (default off): a question with more options than that (noul 2, choice its
  options, score its levels) is not sent to the tier and counts as a tier error there.

Response: today's schema, plus ``tier`` (1-based) and ``tier_model`` on every answer; ``model`` = the cascade id,
``backend`` = "cascade", ``usage`` = the sum over the tiers that were called, ``timing.tiers`` = per tier
``ms``, ``questions`` (sent), ``answered``, ``skipped`` (max_options / unsupported), ``final`` (answers used).

Config (``cascade.json``, see ``CascadeConfig``)::

    {"id": "glue-2", "description": "...",
     "tiers": [{"name": "2b", "model_id": "jev-style-2b-decision-v3", "target": "local:2b-v3"},
               {"name": "jevk5-9b", "model_id": "alibiserikbay/JevK5-9B", "target": "http://127.0.0.1:8090",
                "revision": "v0.3.3", "protocol": "jevk5"}],
     "thresholds": [0.6], "confidence": "normalized_pmax",
     "frozen_utc": "2026-10-03T00:00:00Z", "calibration_sha256": "<64 hex>"}

Named cascades: ``jev_style.cascades.CASCADES`` (``jev-style releases`` lists them). Everywhere a cascade is taken
(``--cascade``, ``JevStyle(cascade=...)``, the eval engine ``cascade:...``) a registry name works as well as a file.

Offline replay: ``select([answer_tier1, answer_tier2, ...], thresholds)`` -> ``Routed(tier, answer)``, where each
entry is the tier's answer dict for one question as recorded (its systemone answer), an error (``TierError``, an
exception, or a ``{"error": ...}`` dict) or ``None`` (not recorded / not asked).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from .adapter import confidence, renormalise
from .schema import (ApiError, ParsedRequest, QuestionSpec, _level_text, compact_json, parse_question,
                     parse_request)

log = logging.getLogger("jev_style.cascade")

CONFIDENCE_RULES = ("normalized_pmax",)
PROTOCOLS = ("systemone", "jevk5")          # jevk5: allebee/jevk5 ``jevk5-serve`` (see ``jevk5_questions``)
TIER_TARGET_KINDS = ("local", "hf", "http", "fake", "jevk5")


class CascadeConfigError(ValueError):
    """A cascade file that does not follow the schema (the message names the field)."""


# ----------------------------------------------------------------------------- routing core (pure)
@dataclass(frozen=True)
class TierError:
    """A tier's failure on one question (the request's error when the whole request failed)."""
    status: int
    code: str
    message: str
    question: str | None = None
    skipped: bool = False           # never sent: more options than the tier's max_options, or unsupported there

    @classmethod
    def from_exception(cls, exc: BaseException) -> "TierError":
        if isinstance(exc, ApiError):
            return cls(exc.status, exc.code, exc.message, exc.question)
        from .client import JevStyleError
        if isinstance(exc, JevStyleError):
            err = exc.body.get("error") if isinstance(exc.body, dict) else None
            if exc.status in (401, 403):            # our key for the tier is wrong: not the caller's problem
                return cls(502, "tier_unauthorized", f"the tier refused this server's credentials: {exc.message}")
            if isinstance(err, dict) and err.get("code"):       # a systemone contract error: keep it
                return cls(exc.status, exc.code, exc.message, exc.question)
            return cls(502, "tier_error", f"HTTP {exc.status}: {exc.message}")
        try:
            import httpx
            if isinstance(exc, httpx.TimeoutException):
                return cls(504, "tier_timeout", f"{type(exc).__name__}: {exc}")
            if isinstance(exc, httpx.TransportError):
                return cls(502, "tier_unavailable", f"{type(exc).__name__}: {exc}")
        except ImportError:  # pragma: no cover - httpx is a dependency
            pass
        if isinstance(exc, OSError):
            return cls(502, "tier_unavailable", f"{type(exc).__name__}: {exc}")
        return cls(500, "tier_failed", f"{type(exc).__name__}: {exc}")

    @classmethod
    def coerce(cls, value: Any) -> "TierError":
        """A recorded error (TierError, exception or {"error": ...} body) as a TierError."""
        if isinstance(value, TierError):
            return value
        if isinstance(value, BaseException):
            return cls.from_exception(value)
        err = value.get("error") if isinstance(value, dict) else None
        status = int(value.get("status", 500)) if isinstance(value, dict) else 500
        if isinstance(err, dict):
            return cls(status, str(err.get("code", "tier_error")), str(err.get("message", "")), err.get("question"))
        return cls(status, "tier_error", str(err if err is not None else value))


def is_error(value: Any) -> bool:
    """A tier result that is an error: TierError, an exception, or an error body {"error": ...} (without "type")."""
    return (isinstance(value, (TierError, BaseException))
            or (isinstance(value, dict) and "error" in value and "type" not in value))


class Normalized(NamedTuple):
    label: str                          # top option ("true" / "false" for noul, the level index for score)
    probabilities: dict[str, float]     # renormalised, in the answer's own key order (noul: false, true)
    confidence: float                   # normalized p_max (see the module docstring)


def normalize(answer: Mapping[str, Any]) -> Normalized:
    """Any tier's systemone answer -> (top label, probabilities, confidence), from the probabilities only.
    Raises ValueError for a malformed answer."""
    if not isinstance(answer, Mapping):
        raise ValueError(f"answer must be an object, got {type(answer).__name__}")
    t = answer.get("type")
    if t == "noul":
        if "noul" in answer:
            p = _prob(answer["noul"], "noul")
        else:                                       # a tier that sends {"probabilities": {"false", "true"}}
            probs = answer.get("probabilities")
            if not isinstance(probs, Mapping) or set(probs) != {"false", "true"}:
                raise ValueError("noul answer needs 'noul' or probabilities over 'false' / 'true'")
            p = _renormalised([_prob(probs["false"], "false"), _prob(probs["true"], "true")])[1]
        if p > 1.0 + 1e-9:
            raise ValueError(f"noul must be a probability in [0, 1], got {p!r}")
        p = min(1.0, p)
        return Normalized("true" if p > 0.5 else "false", {"false": 1.0 - p, "true": p}, abs(2.0 * p - 1.0))
    if t in ("choice", "score"):
        probs = answer.get("probabilities")
        if not isinstance(probs, Mapping) or not probs:
            raise ValueError(f"{t} answer needs a non-empty 'probabilities' object")
        names = [str(k) for k in probs]
        p = _renormalised([_prob(v, f"probabilities[{k!r}]") for k, v in probs.items()])
        i = max(range(len(p)), key=p.__getitem__)  # first maximum, as Adapter
        return Normalized(names[i], dict(zip(names, p)), confidence(p))
    raise ValueError(f"answer type must be noul, choice or score (got {t!r})")


def _prob(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where} must be a number, got {v!r}")
    x = float(v)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError(f"{where} must be a finite probability >= 0, got {v!r}")
    return x


def _renormalised(p: list[float]) -> list[float]:
    if sum(p) <= 0.0:
        raise ValueError("probabilities sum to 0")
    return renormalise(p)


def _status(tier: int, result: Any, thresholds: Sequence[float]) -> str:
    """'keep' | 'low' (answered, below the tier's threshold) | 'error' | 'none' (not asked)."""
    if result is None:
        return "none"
    if is_error(result):
        return "error"
    try:
        c = normalize(result).confidence
    except (ValueError, KeyError, TypeError):
        return "error"
    return "keep" if tier >= len(thresholds) or c >= thresholds[tier] else "low"


def keep(tier: int, result: Any, thresholds: Sequence[float]) -> bool:
    """Is ``result`` (tier ``tier``'s answer to one question, 0-based) final? The top tier (``tier ==
    len(thresholds)``) keeps any answer; an error or a missing result is never kept."""
    return _status(tier, result, thresholds) == "keep"


class Routed(NamedTuple):
    tier: int       # 0-based index of the tier whose answer is used (or whose error is surfaced)
    answer: Any     # that tier's answer, or a TierError when every tier failed

    @property
    def ok(self) -> bool:
        return not isinstance(self.answer, TierError)


def option_count(answer: Mapping[str, Any]) -> int:
    return 2 if answer.get("type") == "noul" else len(answer.get("probabilities") or ())


def too_many_options(k: int, limit: int, question: str | None = None, tier: str = "") -> TierError:
    return TierError(422, "too_many_options", f"{k} options; tier {tier or '?'} takes at most {limit} "
                     "(max_options), so it was not asked", question, skipped=True)


def select(per_tier_answers: Sequence[Any], thresholds: Sequence[float], *, question: Any = None,
           max_options: Sequence[int | None] | None = None, n_options: int | None = None) -> Routed:
    """Route one question. ``per_tier_answers[i]`` is tier i's result for it (answer dict, error, or None when not
    asked); ``thresholds[i]`` is tier i's threshold (one per non-top tier). Walks the tiers in order and returns
    the first answer that ``keep`` accepts; if none does (the top tier failed or was not asked), the last answer
    that succeeded; if every tier failed, the error of the lowest tier that ran the question (a ``max_options``
    skip only when nothing ran it).

    For offline replays: ``question`` (the request's question object, or a QuestionSpec) checks every answer
    against it as the live cascade does (type and option keys; a mismatch is a tier error) and gives the option
    count; ``max_options`` (one entry per tier, None = no limit) applies the max_options rule: a tier whose limit
    is below the option count (``n_options``, else from ``question``, else read from an answer) counts as not
    asked. The live CascadeAdapter applies both before routing and passes the skips in as errors."""
    results = list(per_tier_answers)
    n = len(thresholds) + 1
    if not 1 <= len(results) <= n:
        raise ValueError(f"expected 1..{n} tier results for {len(thresholds)} thresholds, got {len(results)}")
    if question is not None:
        spec = question if isinstance(question, QuestionSpec) else parse_question("question", question)
        results = [check_answer(spec, r) if r is not None and not is_error(r) else r for r in results]
        n_options = len(spec.options) if n_options is None else n_options
    if max_options is not None:
        if len(max_options) != n:
            raise ValueError(f"max_options needs one entry per tier ({n}), got {len(max_options)}")
        k = n_options if n_options is not None else next(
            (option_count(r) for r in results if isinstance(r, Mapping) and not is_error(r) and "type" in r), None)
        if k is not None:
            results = [too_many_options(k, m) if m is not None and k > m else r
                       for r, m in zip(results, max_options)]
    last_ok: Routed | None = None
    errors: list[Routed] = []
    for i, r in enumerate(results):
        s = _status(i, r, thresholds)
        if s == "keep":
            return Routed(i, r)
        if s == "low":
            last_ok = Routed(i, r)
        elif s == "error":
            errors.append(Routed(i, TierError.coerce(r) if is_error(r) else _bad_answer(r)))
    if last_ok is not None:
        return last_ok
    if not errors:
        raise ValueError("no tier result to select from (every entry is None)")
    ran = [e for e in errors if not e.answer.skipped]
    return (ran or errors)[0]


def _bad_answer(answer: Any, question: str | None = None) -> TierError:
    try:
        normalize(answer)
        why = "answer does not match the question"
    except (ValueError, KeyError, TypeError) as e:
        why = str(e)
    return TierError(502, "tier_bad_answer", f"malformed answer: {why}", question)


def canonical(q: QuestionSpec, norm: Normalized) -> dict:
    """Today's answer schema (what ``Adapter`` returns) from a normalized tier answer."""
    if q.type == "noul":
        return {"type": "noul", "noul": norm.probabilities["true"]}
    probs = {n: norm.probabilities[n] for n in q.options}
    if q.type == "choice":
        return {"type": "choice", "choice": norm.label, "confidence": norm.confidence, "probabilities": probs}
    return {"type": "score", "score": float(sum(i * v for i, v in enumerate(probs.values()))),
            "confidence": norm.confidence, "legend": dict(q.legend), "probabilities": probs}


def check_answer(q: QuestionSpec, answer: Any) -> Any:
    """The tier's answer if it fits question ``q`` (type and option keys), else a TierError."""
    if not isinstance(answer, Mapping):
        return TierError(502, "tier_bad_answer", "the tier returned no answer for this question", q.id)
    try:
        norm = normalize(answer)
    except (ValueError, KeyError, TypeError):
        return _bad_answer(answer, q.id)
    if answer.get("type") != q.type:
        return TierError(502, "tier_bad_answer", f"answer type {answer.get('type')!r}, question type {q.type!r}",
                         q.id)
    if q.type != "noul" and set(norm.probabilities) != set(q.options):
        return TierError(502, "tier_bad_answer", f"answer options {sorted(norm.probabilities)} do not match the "
                         f"question's {sorted(q.options)}", q.id)
    return answer


# ----------------------------------------------------------------------------- tier protocols
def jevk5_questions(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict]:
    """Questions as ``jevk5-serve`` (allebee/jevk5 0.3.x) reads them. It renders each option as ``"key: description"``
    with Python ``str()``, so structured descriptions are sent as compact JSON text and a score level
    ``{"label", "description"}`` as ``"label: description"`` (the text Jev-Style's own renderer uses); other
    fields are dropped (jevk5 ignores them). ``instructions`` and the state go through unchanged."""
    out = {}
    for qid, q in questions.items():
        t, crit = q.get("type"), q.get("criteria")
        if t in ("noul", "choice") and isinstance(crit, Mapping):
            crit = {k: compact_json(v) if isinstance(v, (dict, list)) else v for k, v in crit.items()}
        elif t == "score" and isinstance(crit, list):
            crit = [_text(_level_text(level)) for level in crit]
        out[qid] = {"type": t, "instructions": q.get("instructions")}
        if crit is not None:
            out[qid]["criteria"] = crit
    return out


def _text(v: Any) -> Any:
    return compact_json(v) if isinstance(v, (dict, list)) else v


@dataclass
class Tier:
    """One cascade tier. ``engine`` has ``decide(state, questions, model=None) -> response body`` and raises
    ``JevStyleError`` / ``ApiError`` (a ``jev_style.JevStyle`` client: in-process or HTTP)."""
    name: str
    model_id: str
    engine: Any
    max_options: int | None = None
    protocol: str = "systemone"
    target: str | None = None
    revision: str | None = None

    def refuses(self, q: QuestionSpec) -> TierError | None:
        """Why ``q`` is not sent to this tier (counts as a tier error there), or None."""
        k = len(q.options)
        if self.max_options is not None and k > self.max_options:
            return too_many_options(k, self.max_options, q.id, self.name)
        if self.protocol == "jevk5" and q.type == "choice" and k < 2:   # jevk5-serve answers 400 for the request
            return TierError(422, "unsupported_question", f"tier {self.name} (jevk5) needs at least two choice "
                             "options, so it was not asked", q.id, skipped=True)
        return None

    def decide(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict:
        qs = jevk5_questions(questions) if self.protocol == "jevk5" else dict(questions)
        return self.engine.decide(state, qs, model=self.model_id)

    @property
    def _adapter(self) -> Any:
        return getattr(self.engine, "_adapter", None)

    @property
    def backend(self) -> str:
        a = self._adapter
        return a.backend if a is not None else ("http" if getattr(self.engine, "base_url", None) else "custom")

    @property
    def max_len(self) -> int | None:
        return getattr(self._adapter, "max_len", None)

    @property
    def head_max(self) -> int | None:
        return getattr(self._adapter, "head_max", None)

    @property
    def untrained(self) -> bool:
        a = self._adapter
        return bool(a.release_info().get("untrained")) if a is not None and hasattr(a, "release_info") else False


# ----------------------------------------------------------------------------- the adapter
class CascadeAdapter:
    """The ``Adapter`` surface (``systemone``, ``models``, ``release_info``, ``model_id``, ``backend``, ``max_len``,
    ``head_max``) over a list of tiers. Thread-safe as long as the tier engines are (JevStyle clients are)."""

    def __init__(self, tiers: Sequence[Tier], thresholds: Sequence[float], *, cascade_id: str,
                 description: str = "", config: "CascadeConfig | None" = None):
        if len(tiers) < 2:
            raise ValueError("a cascade needs at least two tiers")
        if len(thresholds) != len(tiers) - 1:
            raise ValueError(f"{len(tiers)} tiers need {len(tiers) - 1} thresholds, got {len(thresholds)}")
        for t in thresholds:
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0.0 <= t <= 1.0:
                raise ValueError(f"thresholds must be numbers in [0, 1], got {t!r}")
        self.tiers = list(tiers)
        self.thresholds = [float(t) for t in thresholds]
        self.model_id = cascade_id
        self.description = description or f"Cascade {' -> '.join(t.model_id for t in self.tiers)}"
        self.config = config
        self.release_date = None

    @classmethod
    def from_config(cls, config: "CascadeConfig | Mapping[str, Any] | str | Path", *, backend: str = "auto",
                    **load_kw: Any) -> "CascadeAdapter":
        """Build every tier of a cascade (a registry name, file, dict or CascadeConfig, see ``resolve_config``).
        ``backend`` and ``load_kw`` (device, dtype, precision, quant, cuda_graphs) apply to ``local`` / ``hf`` tiers
        that do not say otherwise."""
        cfg = resolve_config(config)
        tiers = [build_tier(t, backend=backend, **load_kw) for t in cfg.tiers]
        return cls(tiers, cfg.thresholds, cascade_id=cfg.id, description=cfg.description, config=cfg)

    # -- Adapter surface -----------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return "cascade"

    @property
    def max_len(self) -> int | None:
        return self.tiers[0].max_len          # the entry tier's budget: where the declared routing applies

    @property
    def head_max(self) -> int | None:
        return self.tiers[0].head_max

    def systemone(self, body: Any) -> dict:
        t0 = time.perf_counter()
        if isinstance(body, ParsedRequest):
            raise TypeError("CascadeAdapter.systemone needs the request body (it forwards the questions as sent)")
        parsed = parse_request(body)
        raw = body["questions"]
        specs = {q.id: q for q in parsed.questions}
        n = len(self.tiers)
        results: dict[str, list[Any]] = {qid: [None] * n for qid in specs}
        usage: dict[str, Any] = {"input_tokens": 0, "state_tokens": 0, "output_tokens": 0}
        stats = [{"tier": i + 1, "name": t.name, "model": t.model_id, "ms": 0.0, "questions": 0, "answered": 0,
                  "skipped": 0, "final": 0, "input_tokens": 0} for i, t in enumerate(self.tiers)]
        pending = list(specs)
        for i, tier in enumerate(self.tiers):
            if not pending:
                break
            send = []
            for qid in pending:
                refusal = tier.refuses(specs[qid])
                if refusal is not None:
                    results[qid][i] = refusal
                    stats[i]["skipped"] += 1
                else:
                    send.append(qid)
            if send:
                t = time.perf_counter()
                got, used = self._ask(tier, parsed.state, {qid: raw[qid] for qid in send}, specs)
                stats[i].update(ms=round((time.perf_counter() - t) * 1000, 1), questions=len(send),
                                answered=sum(not is_error(got[qid]) for qid in send),
                                input_tokens=int(used.get("input_tokens", 0) or 0))
                for k, v in used.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        usage[k] = usage.get(k, 0) + v
                for qid in send:
                    results[qid][i] = got[qid]
            pending = [qid for qid in pending if not keep(i, results[qid][i], self.thresholds)]
        answers = {}
        for qid, q in specs.items():
            routed = select(results[qid], self.thresholds)
            if not routed.ok:
                raise self._failure(results[qid], routed)
            stats[routed.tier]["final"] += 1
            answers[qid] = {**canonical(q, normalize(routed.answer)), "tier": routed.tier + 1,
                            "tier_model": self.tiers[routed.tier].model_id}
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"model": self.model_id, "answers": answers, "usage": usage, "latency_ms": total_ms,
                "timing": {"total_ms": total_ms, "tiers": stats}, "backend": self.backend}

    def _ask(self, tier: Tier, state: Any, questions: dict[str, Any],
             specs: dict[str, QuestionSpec]) -> tuple[dict[str, Any], dict]:
        """One request to one tier -> ({qid: answer | TierError}, usage)."""
        try:
            resp = tier.decide(state, questions)
        except Exception as e:  # noqa: BLE001 - any tier failure escalates (rule), it is logged
            err = TierError.from_exception(e)
            (log.exception if err.code == "tier_failed" else log.warning)(
                "cascade %s: tier %s failed on %d question(s): %s %s", self.model_id, tier.name, len(questions),
                err.code, err.message)
            return {qid: err for qid in questions}, {}
        answers = resp.get("answers") if isinstance(resp, Mapping) else None
        if not isinstance(answers, Mapping):
            err = TierError(502, "tier_bad_answer", "the tier's response has no 'answers' object")
            return {qid: err for qid in questions}, {}
        used = resp.get("usage") if isinstance(resp.get("usage"), Mapping) else {}
        return {qid: check_answer(specs[qid], answers.get(qid)) for qid in questions}, dict(used)

    def _failure(self, results: list[Any], routed: Routed) -> ApiError:
        err: TierError = routed.answer
        tier_errors = [{"tier": i + 1, "tier_model": self.tiers[i].model_id, "status": e.status, "code": e.code,
                        "message": e.message} for i, r in enumerate(results) if is_error(r)
                       for e in (TierError.coerce(r),)]
        return ApiError(err.status, err.code, err.message, err.question, tier=routed.tier + 1,
                        tier_model=self.tiers[routed.tier].model_id, tier_errors=tier_errors)

    def tier_info(self) -> list[dict]:
        return [{"tier": i + 1, "name": t.name, "model": t.model_id, "target": t.target, "revision": t.revision,
                 "protocol": t.protocol, "max_options": t.max_options, "backend": t.backend,
                 "threshold": self.thresholds[i] if i < len(self.thresholds) else None}
                for i, t in enumerate(self.tiers)]

    def models(self) -> dict:
        entry = {"id": self.model_id, "context_tokens": self.max_len, "head_max_tokens": self.head_max,
                 "backend": self.backend, "tiers": self.tier_info(), **self._frozen()}
        return {"object": "list", "data": [entry],
                "models": [{"name": self.model_id, "release_date": self.release_date,
                            "description": self.description}]}

    def release_info(self) -> dict:
        return {"model_name": self.model_id, "release_kind": "cascade",
                "untrained": any(t.untrained for t in self.tiers), "acceptance_status": None,
                "tiers": [t.model_id for t in self.tiers], **self._frozen()}

    def _frozen(self) -> dict:
        c = self.config
        return {"confidence": "normalized_pmax", "frozen_utc": c.frozen_utc if c else None,
                "calibration_sha256": c.calibration_sha256 if c else None,
                "config_sha256": c.sha256 if c else None}


# ----------------------------------------------------------------------------- config
_TOP_KEYS = {"id", "description", "tiers", "thresholds", "confidence", "frozen_utc", "calibration_sha256", "$schema"}
_TOP_REQUIRED = ("id", "description", "tiers", "thresholds")
_TIER_KEYS = {"name", "model_id", "target", "revision", "max_options", "protocol", "api_key_env", "timeout_s", "dtype",
              "verify"}
TIER_DTYPES = ("float32", "bfloat16")
_TIER_REQUIRED = ("name", "model_id", "target")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TierConfig:
    name: str                           # short, unique within the cascade ("2b", "jevk5-9b")
    model_id: str                       # reported as tier_model; sent as "model" to the tier
    target: str                         # local[:release][:backend] | hf:<repo> | jevk5:<repo or folder>
                                        # | http(s)://host:port | fake
    revision: str | None = None         # local / hf / jevk5: the Hub revision to load; http: recorded only
    max_options: int | None = None      # questions with more options are not sent to this tier
    protocol: str = "systemone"         # "jevk5": shape questions for JevK5 (see jevk5_questions); jevk5: targets
                                        # always use it
    api_key_env: str | None = None      # http: bearer token read from this environment variable
    timeout_s: float | None = None      # http: request timeout (default 120 s)
    dtype: str | None = None            # local / hf torch tiers: "float32" | "bfloat16" (overrides --dtype)
    verify: bool | None = None          # local / hf: the runtime's manifest check (default off); jevk5: SHA256SUMS
                                        # (default on for the pinned release, see jevk5_engine)


@dataclass(frozen=True)
class CascadeConfig:
    """A cascade file. Required: id, description, tiers (>= 2), thresholds (len(tiers) - 1 numbers in [0, 1]).
    Optional: confidence (only "normalized_pmax"), frozen_utc (ISO 8601, UTC), calibration_sha256 (64 hex).
    Unknown keys are refused, so a typo cannot silently change a frozen cascade."""
    id: str
    description: str
    tiers: tuple[TierConfig, ...]
    thresholds: tuple[float, ...]
    confidence: str = "normalized_pmax"
    frozen_utc: str | None = None
    calibration_sha256: str | None = None
    sha256: str | None = field(default=None, compare=False)     # of the file it was read from

    @classmethod
    def from_dict(cls, d: Any, sha256: str | None = None) -> "CascadeConfig":
        if not isinstance(d, Mapping):
            raise CascadeConfigError("cascade config must be a JSON object")
        _keys(d, _TOP_KEYS, _TOP_REQUIRED, "cascade")
        cid = _string(d["id"], "id")
        if not isinstance(d["description"], str):
            raise CascadeConfigError("description must be a string")
        tiers_raw = d["tiers"]
        if not isinstance(tiers_raw, list) or len(tiers_raw) < 2:
            raise CascadeConfigError("tiers must be an array of at least two tiers (smallest first)")
        tiers = tuple(_tier(t, i) for i, t in enumerate(tiers_raw))
        names = [t.name for t in tiers]
        if len(set(names)) != len(names):
            raise CascadeConfigError(f"tier names must be unique, got {names}")
        th = d["thresholds"]
        if not isinstance(th, list) or len(th) != len(tiers) - 1:
            raise CascadeConfigError(f"thresholds must be an array of {len(tiers) - 1} number(s): one per tier "
                                     "below the top")
        for i, t in enumerate(th):
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 <= t <= 1:
                raise CascadeConfigError(f"thresholds[{i}] must be a number in [0, 1], got {t!r}")
        rule = d.get("confidence", "normalized_pmax")
        if rule not in CONFIDENCE_RULES:
            raise CascadeConfigError(f"confidence must be one of {list(CONFIDENCE_RULES)}, got {rule!r}")
        frozen = d.get("frozen_utc")
        if frozen is not None:
            _utc(frozen)
        sha = d.get("calibration_sha256")
        if sha is not None and not (isinstance(sha, str) and _SHA256.match(sha)):
            raise CascadeConfigError("calibration_sha256 must be 64 lowercase hex characters")
        return cls(cid, d["description"], tiers, tuple(float(t) for t in th), rule, frozen, sha, sha256)

    def to_dict(self) -> dict:
        tiers = []
        for t in self.tiers:
            row = {"name": t.name, "model_id": t.model_id, "target": t.target}
            for k in ("revision", "max_options", "api_key_env", "timeout_s", "dtype", "verify"):
                if getattr(t, k) is not None:
                    row[k] = getattr(t, k)
            if t.protocol != "systemone":
                row["protocol"] = t.protocol
            tiers.append(row)
        out = {"id": self.id, "description": self.description, "tiers": tiers, "thresholds": list(self.thresholds),
               "confidence": self.confidence}
        for k in ("frozen_utc", "calibration_sha256"):
            if getattr(self, k) is not None:
                out[k] = getattr(self, k)
        return out


def load_config(path: str | Path) -> CascadeConfig:
    p = Path(path).expanduser()
    try:
        data = p.read_bytes()
    except OSError as e:
        raise CascadeConfigError(f"cannot read cascade file {p}: {e}") from None
    try:
        d = json.loads(data)
    except ValueError as e:
        raise CascadeConfigError(f"{p} is not valid JSON: {e}") from None
    try:
        return CascadeConfig.from_dict(d, sha256=hashlib.sha256(data).hexdigest())
    except CascadeConfigError as e:
        raise CascadeConfigError(f"{p}: {e}") from None


def _keys(d: Mapping, allowed: set, required: Sequence[str], where: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise CascadeConfigError(f"{where}: unknown key(s) {unknown} (allowed: {sorted(allowed - {'$schema'})})")
    missing = [k for k in required if k not in d]
    if missing:
        raise CascadeConfigError(f"{where}: missing {missing}")


def _string(v: Any, where: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise CascadeConfigError(f"{where} must be a non-empty string")
    return v


def _utc(v: Any) -> None:
    try:
        ts = datetime.fromisoformat(v[:-1] + "+00:00" if isinstance(v, str) and v.endswith("Z") else v)
    except (TypeError, ValueError):
        raise CascadeConfigError(f"frozen_utc must be an ISO 8601 timestamp, got {v!r}") from None
    if ts.utcoffset() != timedelta(0):
        raise CascadeConfigError(f"frozen_utc must be in UTC (Z or +00:00), got {v!r}")


def _tier(t: Any, i: int) -> TierConfig:
    where = f"tiers[{i}]"
    if not isinstance(t, Mapping):
        raise CascadeConfigError(f"{where} must be an object")
    _keys(t, _TIER_KEYS, _TIER_REQUIRED, where)
    name, model_id, target = (_string(t[k], f"{where}.{k}") for k in ("name", "model_id", "target"))
    from .client import parse_target
    try:
        kind, kw = parse_target(target)
    except ValueError as e:
        raise CascadeConfigError(f"{where}.target: {e}") from None
    if kind not in TIER_TARGET_KINDS:
        raise CascadeConfigError(f"{where}.target: a tier must be local[:release][:backend], hf:<repo>, "
                                 f"jevk5:<repo or folder>, http(s)://... or fake (got {kind!r})")
    if kind == "local" and "release" in kw:
        from .models import get_release
        try:
            get_release(kw["release"])
        except ValueError as e:
            raise CascadeConfigError(f"{where}.target: {e}") from None
    rev = t.get("revision")
    if rev is not None:
        _string(rev, f"{where}.revision")
    mo = t.get("max_options")
    if mo is not None and (isinstance(mo, bool) or not isinstance(mo, int) or mo < 1):
        raise CascadeConfigError(f"{where}.max_options must be a positive integer or null, got {mo!r}")
    proto = t.get("protocol", "jevk5" if kind == "jevk5" else "systemone")
    if proto not in PROTOCOLS:
        raise CascadeConfigError(f"{where}.protocol must be one of {list(PROTOCOLS)}, got {proto!r}")
    if kind == "jevk5" and proto != "jevk5":
        raise CascadeConfigError(f"{where}.protocol: a jevk5: target answers as jevk5-serve does, so its protocol is "
                                 f"'jevk5' (got {proto!r})")
    key_env, timeout = t.get("api_key_env"), t.get("timeout_s")
    if (key_env is not None or timeout is not None) and kind != "http":
        raise CascadeConfigError(f"{where}: api_key_env / timeout_s only apply to http(s) targets")
    if key_env is not None:
        _string(key_env, f"{where}.api_key_env")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0):
        raise CascadeConfigError(f"{where}.timeout_s must be a positive number, got {timeout!r}")
    dtype = t.get("dtype")
    if dtype is not None:
        if kind not in ("local", "hf"):
            raise CascadeConfigError(f"{where}.dtype only applies to local / hf tiers")
        if dtype not in TIER_DTYPES:
            raise CascadeConfigError(f"{where}.dtype must be one of {list(TIER_DTYPES)}, got {dtype!r}")
    verify = t.get("verify")
    if verify is not None:
        if kind not in ("local", "hf", "jevk5"):
            raise CascadeConfigError(f"{where}.verify only applies to local / hf / jevk5 tiers")
        if not isinstance(verify, bool):
            raise CascadeConfigError(f"{where}.verify must be true, false or null, got {verify!r}")
    return TierConfig(name, model_id, target, rev, mo, proto, key_env, float(timeout) if timeout else None, dtype,
                      verify)


def build_tier(cfg: TierConfig, *, backend: str = "auto", **load_kw: Any) -> Tier:
    """Load / connect one tier through the shared engine-target resolver (``jev_style.client.from_target``)."""
    from .client import from_target, parse_target
    kind, _ = parse_target(cfg.target)
    kw: dict[str, Any] = {}
    if kind == "http":
        if cfg.api_key_env:
            key = os.environ.get(cfg.api_key_env, "").strip()
            if not key:
                raise CascadeConfigError(f"tier {cfg.name}: environment variable {cfg.api_key_env} is empty or unset")
            kw["api_key"] = key
        if cfg.timeout_s:
            kw["timeout"] = cfg.timeout_s
    elif kind in ("local", "hf"):
        kw.update(load_kw)
        if cfg.revision:
            kw["revision"] = cfg.revision
        if cfg.dtype:
            kw["dtype"] = cfg.dtype
        if cfg.verify is not None:
            kw["verify"] = cfg.verify
    elif kind == "jevk5":                   # jevk5-serve's own settings: the loader defaults do not apply
        kw.update(revision=cfg.revision, verify=cfg.verify)
    engine = from_target(cfg.target, backend=backend, **kw)
    return Tier(cfg.name, cfg.model_id, engine, cfg.max_options, cfg.protocol, cfg.target, cfg.revision)


# ----------------------------------------------------------------------------- named cascades, downloads
def _registry_key(spec: Any) -> str | None:
    if not isinstance(spec, str):
        return None
    from .cascades import CASCADES
    key = spec.strip().lower()
    return key if key in CASCADES else None


def resolve_config(spec: "CascadeConfig | Mapping[str, Any] | str | Path") -> CascadeConfig:
    """A cascade from a CascadeConfig, a dict, a registry name (``jev_style.cascades.CASCADES``; a name wins over a
    file of the same name, use ``./name`` for the file) or a path to a cascade file. A named cascade must be frozen:
    one whose threshold or frozen_utc is still a placeholder in this version is refused."""
    if isinstance(spec, CascadeConfig):
        return spec
    if isinstance(spec, Mapping):
        return CascadeConfig.from_dict(spec)
    key = _registry_key(spec)
    if key is None:
        try:
            return load_config(spec)
        except CascadeConfigError as e:
            from .cascades import CASCADES
            p = Path(spec).expanduser()
            if not p.exists() and p.suffix != ".json" and len(p.parts) == 1:
                raise CascadeConfigError(f"{e} ({spec!r} is not a named cascade either; known: "
                                         f"{', '.join(CASCADES)})") from None
            raise
    from . import cascades
    missing = cascades.placeholders(key)
    if missing:
        verb = "is" if len(missing) == 1 else "are"
        raise CascadeConfigError(f"cascade {key} is not frozen in jev-style {_version()}: {', '.join(missing)} {verb} "
                                 "still a placeholder in jev_style/cascades.py, so it is not loaded (its tiers can be "
                                 f"downloaded: jev-style download --cascade {key})")
    d = cascades.get(key)
    try:
        cfg = CascadeConfig.from_dict(d)
    except CascadeConfigError as e:
        raise CascadeConfigError(f"cascade {key}: {e}") from None
    canon = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return replace(cfg, sha256=hashlib.sha256(canon.encode()).hexdigest())   # of its canonical JSON


def resolve_tiers(spec: "CascadeConfig | Mapping[str, Any] | str | Path") -> tuple[TierConfig, ...]:
    """The tiers of a cascade, without requiring it to be frozen (for downloads ahead of time)."""
    key = _registry_key(spec)
    if key is None:
        return resolve_config(spec).tiers
    from . import cascades
    return tuple(_tier(t, i) for i, t in enumerate(cascades.get(key)["tiers"]))


def download_tiers(spec: "CascadeConfig | Mapping[str, Any] | str | Path", *, backend: str = "auto",
                   precision: str = "bf16", quant: str = "Q8_0") -> list[tuple[TierConfig, Path | None]]:
    """Fetch every tier's files ahead of time: local / hf tiers the build that would load (``backend`` unless the
    target names one), jevk5 tiers the snapshot, checked as loading checks it (SHA256SUMS for the pinned release,
    the calibration config always; the jevk5 package is not needed). http / fake tiers have nothing to download
    (None). Returns (tier, folder) pairs in tier order."""
    from .client import parse_target
    from .models import build_for_repo, download
    out: list[tuple[TierConfig, Path | None]] = []
    for t in resolve_tiers(spec):
        kind, kw = parse_target(t.target)
        if kind == "local":
            folder = download(kw.get("backend", backend), precision=precision, quant=quant, revision=t.revision,
                              release=kw.get("release"))
        elif kind == "hf":
            folder = download(build=build_for_repo(kw["repo"]), precision=precision, quant=quant,
                              revision=t.revision)
        elif kind == "jevk5":
            from .jevk5_engine import download as download_jevk5
            folder = download_jevk5(kw["jevk5"], revision=t.revision, verify=t.verify)
        else:
            folder = None
        out.append((t, folder))
    return out


def _version() -> str:
    from . import __version__
    return __version__
