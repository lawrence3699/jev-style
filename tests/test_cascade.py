"""The confidence cascade, on scripted and fake tiers (no model, no network)."""
import json
import math
import random

import httpx
import pytest
from fastapi.testclient import TestClient

from jev_style import JevStyle, JevStyleError, choice, noul, score
from jev_style.cascade import (CascadeAdapter, CascadeConfig, CascadeConfigError, Routed, Tier, TierError,
                               jevk5_questions, keep, load_config, normalize, select)
from jev_style.client import parse_target
from jev_style.schema import ApiError
from jev_style.server import create_app

STATE = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."


def answer(q: dict, v):
    """A tier answer for question q. noul: v = P(true); choice: v = {option: p}; score: v = [p0, p1, ...]. The
    ``confidence`` field is deliberately wrong: the cascade must compute its own."""
    t = q["type"]
    if t == "noul":
        return {"type": "noul", "noul": v, "confidence": 0.999}
    if t == "choice":
        return {"type": "choice", "choice": max(v, key=v.get), "confidence": 0.0, "probabilities": dict(v)}
    probs = {str(i): p for i, p in enumerate(v)}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(v)), "confidence": 0.0, "probabilities": probs}


class Scripted:
    """A tier engine: ``table[qid]`` is the answer value (see ``answer``) or an exception (fails the request);
    ``fail`` fails every request. Records each call."""

    def __init__(self, table=None, fail=None):
        self.table, self.fail, self.calls = table or {}, fail, []

    def decide(self, state, questions, model=None):
        self.calls.append({"state": state, "questions": dict(questions), "model": model})
        if self.fail is not None:
            raise self.fail
        out = {}
        for qid, q in questions.items():
            v = self.table[qid]
            if isinstance(v, Exception):
                raise v
            out[qid] = v(q) if callable(v) else answer(q, v)
        return {"model": model, "answers": out, "usage": {"input_tokens": 10 * len(questions), "output_tokens": 0}}


def cascade(*engines, thresholds, max_options=None, protocols=None):
    tiers = [Tier(f"t{i + 1}", f"model-{i + 1}", e, (max_options or [None] * len(engines))[i],
                  (protocols or ["systemone"] * len(engines))[i]) for i, e in enumerate(engines)]
    return CascadeAdapter(tiers, thresholds, cascade_id="test-cascade")


QS = {"n": noul("Does this need urgent human attention?"),
      "c": choice("Which team?", {"returns": "exchanges", "shipping": "delays", "billing": "charges"}),
      "s": score("How frustrated?", ["Calm", {"label": "Frustrated", "description": "clearly unhappy"}, "Very angry"])}


# ----------------------------------------------------------------------------- normalize / select (pure)
def test_normalized_confidence_per_type():
    n = normalize({"type": "noul", "noul": 0.8, "confidence": 0.1})
    assert n.label == "true" and n.confidence == pytest.approx(0.6) and n.probabilities == pytest.approx(
        {"false": 0.2, "true": 0.8})
    assert normalize({"type": "noul", "noul": 0.3}).confidence == abs(2 * 0.3 - 1) and \
        normalize({"type": "noul", "noul": 0.3}).label == "false"
    c = normalize({"type": "choice", "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}, "confidence": 0.99})
    assert c.label == "a" and c.confidence == pytest.approx((3 * 0.5 - 1) / 2)
    s = normalize({"type": "score", "probabilities": {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.7}})
    assert s.label == "3" and s.confidence == pytest.approx((4 * 0.7 - 1) / 3)
    # renormalised once, like Adapter; noul given as probabilities also works
    assert normalize({"type": "choice", "probabilities": {"a": 2.0, "b": 2.0}}).confidence == 0.0
    assert normalize({"type": "noul", "probabilities": {"false": 0.1, "true": 0.9}}).confidence == pytest.approx(0.8)


@pytest.mark.parametrize("bad", [{"type": "noul"}, {"type": "choice", "probabilities": {}},
                                 {"type": "choice", "probabilities": {"a": float("nan"), "b": 0.5}},
                                 {"type": "score", "probabilities": {"0": -0.1, "1": 1.1}},
                                 {"type": "choice", "probabilities": {"a": 0, "b": 0}}, {"type": "maybe"}, "x"])
def test_malformed_answers_are_errors(bad):
    with pytest.raises(ValueError):
        normalize(bad)
    assert not keep(0, bad, [0.5])
    assert not select([bad], []).ok


@pytest.mark.parametrize("q,low,high,tau", [
    (QS["n"], 0.69, 0.71, 0.4),                                         # |2p - 1|: 0.38 / 0.42
    (QS["c"], {"returns": 0.5, "shipping": 0.3, "billing": 0.2},
     {"returns": 0.7, "shipping": 0.2, "billing": 0.1}, 0.5),
    (QS["s"], [0.2, 0.3, 0.5], [0.05, 0.05, 0.9], 0.5),
])
def test_select_keeps_or_escalates_per_type(q, low, high, tau):
    top = answer(q, high)
    assert select([answer(q, high), top], [tau]) == Routed(0, answer(q, high))
    assert select([answer(q, low), top], [tau]).tier == 1
    # the answer's own "confidence" field is ignored (Scripted-style answers carry 0.999 / 0.0)
    assert keep(0, answer(q, high), [tau]) and not keep(0, answer(q, low), [tau])


def test_threshold_is_inclusive_and_top_keeps_everything():
    a = {"type": "choice", "probabilities": {"a": 0.75, "b": 0.25}}            # confidence exactly 0.5
    assert keep(0, a, [0.5]) and not keep(0, a, [0.5000001])
    assert keep(1, {"type": "noul", "noul": 0.5}, [0.9])                      # top tier: confidence 0, still final


def test_select_errors_and_fallback():
    lo, hi = {"type": "noul", "noul": 0.55}, {"type": "noul", "noul": 0.99}
    budget = TierError(422, "input_budget_exceeded", "too long")
    down = TierError(502, "tier_unavailable", "connection refused")
    assert select([budget, hi], [0.5]) == Routed(1, hi)                       # error escalates
    assert select([lo, down], [0.5]) == Routed(0, lo)                         # top failed: last success
    assert select([lo, lo, down], [0.5, 0.5]) == Routed(1, lo)                # ... the LAST tier that succeeded
    assert select([lo, budget, down], [0.5, 0.5]) == Routed(0, lo)
    r = select([budget, down], [0.5])                                         # all failed: lowest tier's error
    assert not r.ok and r.tier == 0 and r.answer.code == "input_budget_exceeded"
    # recorded errors may be exceptions or error bodies; None = not asked
    assert select([RuntimeError("x"), {"error": {"code": "invalid_question", "message": "m"}}], [0.5]).answer.code \
        == "tier_failed"
    assert select([lo, None], [0.5]) == Routed(0, lo)
    with pytest.raises(ValueError):
        select([None, None], [0.5])
    with pytest.raises(ValueError):
        select([lo, lo, lo], [0.5])


def test_select_max_options_offline():
    a = {"type": "choice", "probabilities": {k: 0.2 for k in "abcde"}}
    top = {"type": "choice", "probabilities": {"a": 0.9, **{k: 0.025 for k in "bcde"}}}
    # tier 1 would keep it (threshold 0) but takes at most 4 options
    assert select([a, top], [0.0]) == Routed(0, a)
    assert select([a, top], [0.0], max_options=[4, None]) == Routed(1, top)
    assert select([a, top], [0.0], max_options=[4, None], n_options=3) == Routed(0, a)
    r = select([a, top], [0.0], max_options=[4, 4])
    assert not r.ok and r.answer.code == "too_many_options"


# ----------------------------------------------------------------------------- the live adapter
def test_one_request_per_tier_with_only_the_uncertain_questions():
    t1 = Scripted({"n": 0.95, "c": {"returns": 0.4, "shipping": 0.35, "billing": 0.25}, "s": [0.1, 0.1, 0.8]})
    t2 = Scripted({"n": 0.1, "c": {"returns": 0.1, "shipping": 0.85, "billing": 0.05}, "s": [0.3, 0.4, 0.3]})
    body = {"state": STATE, "questions": QS, "model": "ignored"}
    out = cascade(t1, t2, thresholds=[0.6]).systemone(body)
    assert len(t1.calls) == 1 and set(t1.calls[0]["questions"]) == {"n", "c", "s"}
    assert len(t2.calls) == 1 and set(t2.calls[0]["questions"]) == {"c"}
    assert t2.calls[0]["state"] == STATE and t2.calls[0]["questions"]["c"] == QS["c"]       # forwarded as sent
    assert t1.calls[0]["model"] == "model-1" and t2.calls[0]["model"] == "model-2"
    a = out["answers"]
    assert [a[q]["tier"] for q in ("n", "c", "s")] == [1, 2, 1]
    assert a["c"]["tier_model"] == "model-2" and a["c"]["choice"] == "shipping"
    assert a["n"]["noul"] == 0.95 and "confidence" not in a["n"]
    assert a["c"]["confidence"] == pytest.approx((3 * 0.85 - 1) / 2)              # ours, not the tier's 0.0
    assert a["s"]["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert a["s"]["score"] == pytest.approx(0.1 + 2 * 0.8) and a["s"]["confidence"] == pytest.approx(0.7)
    assert out["model"] == "test-cascade" and out["backend"] == "cascade"
    assert out["usage"] == {"input_tokens": 40, "state_tokens": 0, "output_tokens": 0}
    tiers = out["timing"]["tiers"]
    assert [(t["questions"], t["answered"], t["final"]) for t in tiers] == [(3, 3, 2), (1, 1, 1)]
    assert "total_ms" in out["timing"] and "latency_ms" in out


def test_three_tiers_glue3_routing():
    q = {f"q{i}": noul(f"statement {i}") for i in range(4)}
    t1 = Scripted({"q0": 0.99, "q1": 0.6, "q2": 0.6, "q3": 0.6})
    t2 = Scripted({"q1": 0.97, "q2": 0.55, "q3": 0.6})
    t3 = Scripted({"q2": 0.02, "q3": 0.5})
    out = cascade(t1, t2, t3, thresholds=[0.9, 0.8]).systemone({"state": "s", "questions": q})
    assert [len(t.calls) for t in (t1, t2, t3)] == [1, 1, 1]
    assert set(t2.calls[0]["questions"]) == {"q1", "q2", "q3"} and set(t3.calls[0]["questions"]) == {"q2", "q3"}
    assert {k: v["tier"] for k, v in out["answers"].items()} == {"q0": 1, "q1": 2, "q2": 3, "q3": 3}
    assert out["answers"]["q3"]["noul"] == 0.5                       # top tier is final even at confidence 0


def test_no_call_to_a_tier_with_nothing_to_do():
    t1, t2 = Scripted({"n": 0.99}), Scripted()
    out = cascade(t1, t2, thresholds=[0.5]).systemone({"state": "s", "questions": {"n": QS["n"]}})
    assert t2.calls == [] and out["timing"]["tiers"][1]["questions"] == 0


@pytest.mark.parametrize("exc,code", [
    (ApiError(422, "input_budget_exceeded", "input needs 30000 tokens"), "input_budget_exceeded"),
    (JevStyleError(400, "http_error", "choice criteria must name at least two options",
                   body={"error": "choice criteria must name at least two options"}), "tier_error"),
    (httpx.ConnectError("refused"), "tier_unavailable"),
])
def test_tier_errors_escalate_the_whole_request(exc, code):
    t1 = Scripted(fail=exc)
    t2 = Scripted({"n": 0.2, "c": {"returns": 0.6, "shipping": 0.2, "billing": 0.2}, "s": [0.5, 0.3, 0.2]})
    out = cascade(t1, t2, thresholds=[0.0]).systemone({"state": STATE, "questions": QS})
    assert {a["tier"] for a in out["answers"].values()} == {2} and set(t2.calls[0]["questions"]) == set(QS)
    assert out["timing"]["tiers"][0]["answered"] == 0
    assert TierError.from_exception(exc).code == code


def test_a_malformed_answer_escalates_only_that_question():
    t1 = Scripted({"n": 0.99, "c": lambda q: {"type": "choice", "probabilities": {"returns": 1.0}},   # options missing
                   "s": lambda q: {"type": "noul", "noul": 0.99}})                                   # wrong type
    t2 = Scripted({"c": {"returns": 0.2, "shipping": 0.2, "billing": 0.6}, "s": [0.2, 0.2, 0.6]})
    out = cascade(t1, t2, thresholds=[0.5]).systemone({"state": STATE, "questions": QS})
    assert {k: v["tier"] for k, v in out["answers"].items()} == {"n": 1, "c": 2, "s": 2}


def test_top_tier_failure_falls_back_to_the_last_success():
    t1 = Scripted({"n": 0.6, "c": {"returns": 0.4, "shipping": 0.3, "billing": 0.3}, "s": [0.4, 0.3, 0.3]})
    t2 = Scripted({"n": 0.7, "c": {"returns": 0.2, "shipping": 0.5, "billing": 0.3}, "s": [0.3, 0.3, 0.4]})
    t3 = Scripted(fail=httpx.ReadTimeout("slow"))
    out = cascade(t1, t2, t3, thresholds=[0.9, 0.9]).systemone({"state": STATE, "questions": QS})
    assert {a["tier"] for a in out["answers"].values()} == {2}
    assert out["answers"]["n"]["noul"] == 0.7 and out["answers"]["c"]["choice"] == "shipping"
    two = cascade(Scripted(t1.table), Scripted(fail=httpx.ConnectError("down")), thresholds=[0.9])
    assert {a["tier"] for a in two.systemone({"state": STATE, "questions": QS})["answers"].values()} == {1}


def test_all_tiers_failed_surfaces_the_error_like_adapter():
    t1 = Scripted(fail=ApiError(422, "input_budget_exceeded", "input needs 30000 tokens (nothing truncated)"))
    t2 = Scripted(fail=httpx.ConnectError("refused"))
    a = cascade(t1, t2, thresholds=[0.5])
    with pytest.raises(ApiError) as ei:
        a.systemone({"state": STATE, "questions": QS})
    e = ei.value
    assert (e.status, e.code) == (422, "input_budget_exceeded")
    assert e.message == "input needs 30000 tokens (nothing truncated)"
    body = e.body()["error"]
    assert body["tier"] == 1 and body["tier_model"] == "model-1"
    assert [x["code"] for x in body["tier_errors"]] == ["input_budget_exceeded", "tier_unavailable"]
    r = TestClient(create_app(a)).post("/v1/systemone", json={"state": STATE, "questions": QS})
    assert r.status_code == 422 and r.json()["error"]["code"] == "input_budget_exceeded"


def test_request_validation_happens_before_any_tier():
    t1, t2 = Scripted(), Scripted()
    with pytest.raises(ApiError) as ei:
        cascade(t1, t2, thresholds=[0.5]).systemone({"state": "s", "questions": {"x": {"type": "maybe"}}})
    assert ei.value.code == "invalid_question" and t1.calls == t2.calls == []


def test_max_options_rule_live():
    many = {"m": choice("Which intent?", [f"o{i}" for i in range(20)])}
    qs = {**many, "n": QS["n"], "s": QS["s"]}
    flat = {f"o{i}": 0.05 for i in range(20)}
    t1 = Scripted({"n": 0.99, "s": [0.9, 0.05, 0.05], "m": flat})
    t2 = Scripted({"m": {**flat, "o3": 0.6, "o4": 0.0}})
    out = cascade(t1, t2, thresholds=[0.0], max_options=[16, None]).systemone({"state": "s", "questions": qs})
    assert set(t1.calls[0]["questions"]) == {"n", "s"} and set(t2.calls[0]["questions"]) == {"m"}
    assert out["answers"]["m"]["tier"] == 2 and out["answers"]["m"]["choice"] == "o3"
    assert out["timing"]["tiers"][0]["skipped"] == 1
    # every tier too small: the max_options error is surfaced
    with pytest.raises(ApiError) as ei:
        cascade(Scripted(), Scripted(), thresholds=[0.0], max_options=[16, 16]).systemone(
            {"state": "s", "questions": many})
    assert ei.value.code == "too_many_options" and ei.value.question == "m"


def test_offline_select_matches_the_live_adapter():
    """Record every tier's answer to every question, replay them through select(), and compare with what the live
    cascade returns on the same answers: same tier, same answer."""
    rng = random.Random(7)
    qs, tables = {}, [{}, {}, {}]
    for i in range(60):
        kind = ("noul", "choice", "score")[i % 3]
        if kind == "noul":
            q = noul(f"q{i}")
        elif kind == "choice":
            q = choice(f"q{i}", [f"o{j}" for j in range(rng.randint(2, 20))])
        else:
            q = score(f"q{i}", [f"l{j}" for j in range(rng.randint(2, 10))])
        qs[f"q{i}"] = q
        for t in range(3):
            if rng.random() < 0.1:                     # malformed: wrong option keys / nothing usable
                tables[t][f"q{i}"] = rng.choice([lambda q: {"type": q["type"], "probabilities": {"nope": 1.0}},
                                                 lambda q: {"type": q["type"]}])
            elif kind == "noul":
                tables[t][f"q{i}"] = rng.random()
            else:
                k = len(q["criteria"])
                w = [rng.random() ** 3 for _ in range(k)]
                v = [x / sum(w) for x in w]
                tables[t][f"q{i}"] = dict(zip(q["criteria"], v)) if kind == "choice" else v
    thresholds, max_options = [0.35, 0.5], [12, None, None]
    resp = cascade(*(Scripted(t) for t in tables), thresholds=thresholds, max_options=max_options).systemone(
        {"state": "s", "questions": qs})
    live = resp["answers"]
    assert resp["timing"]["tiers"][0]["skipped"] > 0                     # the max_options rule is exercised
    assert any(t["answered"] < t["questions"] for t in resp["timing"]["tiers"])   # ... and malformed answers
    routed_tiers = set()
    for qid, q in qs.items():
        recorded = []
        for t in tables:
            v = t[qid]
            recorded.append(v(q) if callable(v) else answer(q, v))
        r = select(recorded, thresholds, question=q, max_options=max_options)
        assert r == select(recorded, thresholds, question=q, max_options=max_options,
                           n_options=2 if q["type"] == "noul" else len(q["criteria"]))
        assert r.ok and live[qid]["tier"] == r.tier + 1, qid
        n = normalize(r.answer)
        if q["type"] == "noul":
            assert live[qid]["noul"] == n.probabilities["true"]
        else:
            assert live[qid]["probabilities"] == n.probabilities and live[qid]["confidence"] == n.confidence
        routed_tiers.add(r.tier)
    assert routed_tiers == {0, 1, 2}                 # the sample exercises every tier


# ----------------------------------------------------------------------------- drop-in compatibility
def _fake_config(**over):
    cfg = {"id": "fake-cascade", "description": "two fake tiers",
           "tiers": [{"name": "small", "model_id": "jev-style-fake", "target": "fake"},
                     {"name": "big", "model_id": "jev-style-fake-big", "target": "fake"}],
           "thresholds": [0.5]}
    cfg.update(over)
    return cfg


TICKET = {"state": STATE, "questions": QS}


def test_response_schema_is_a_superset_of_the_single_model_one():
    single = TestClient(create_app(JevStyle(fake=True)._adapter)).post("/v1/systemone", json=TICKET).json()
    app = create_app(CascadeAdapter.from_config(_fake_config(thresholds=[0.0])))
    c = TestClient(app)
    r = c.post("/v1/systemone", json=TICKET)
    assert r.status_code == 200 and r.headers["x-request-id"].startswith("req_")
    body = r.json()
    assert set(single) <= set(body) and set(single["usage"]) <= set(body["usage"])
    for qid, a in single["answers"].items():
        b = body["answers"][qid]
        assert set(b) == set(a) | {"tier", "tier_model"} and b["tier"] == 1 and b["tier_model"] == "jev-style-fake"
        assert list(b) == list(a) + ["tier", "tier_model"]
        for k, v in a.items():
            assert b[k] == (v if k in ("type", "choice", "legend") else pytest.approx(v)), (qid, k)
    assert body["model"] == "fake-cascade" and body["backend"] == "cascade"
    assert body["usage"]["input_tokens"] == single["usage"]["input_tokens"]
    h = c.get("/healthz").json()
    assert h["model"] == "fake-cascade" and h["backend"] == "cascade" and h["release_kind"] == "cascade"
    m = c.get("/v1/models").json()
    assert m["models"][0]["name"] == "fake-cascade" and m["data"][0]["context_tokens"] == 25_600
    assert [t["threshold"] for t in m["data"][0]["tiers"]] == [0.0, None]
    # the client and the agent-approval demo work unchanged on top of it
    out = JevStyle(base_url="http://test", http_client=c).decide(STATE, {"b": noul("About billing?")})
    assert out["answers"]["b"]["tier"] == 1
    assert c.post("/api/agent-approval/check", json={"tool": "Bash", "input": {"command": "ls"}}).status_code == 200


def test_budget_error_through_the_cascade_is_the_adapter_error():
    c = TestClient(create_app(CascadeAdapter.from_config(_fake_config())))
    r = c.post("/v1/systemone", json={"state": "word " * 30_000, "questions": {"x": noul("long?")}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "input_budget_exceeded"
    assert r.json()["error"]["tier"] == 1


# ----------------------------------------------------------------------------- JevK5 (jevk5-serve) tiers
def jevk5_transport(seen):
    from jev_style.fake import jevk5_response

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        try:
            return httpx.Response(200, json=jevk5_response(body))
        except ValueError as e:
            return httpx.Response(400, json={"error": str(e)})
    return httpx.MockTransport(handler)


def test_jevk5_tier_shapes_requests_and_reads_its_answers():
    seen = []
    k5 = JevStyle(base_url="http://k5", http_client=httpx.Client(base_url="http://k5", transport=jevk5_transport(seen)))
    t1 = Scripted({"n": 0.5, "c": {"returns": 0.34, "shipping": 0.33, "billing": 0.33}, "s": [0.34, 0.33, 0.33],
                   "one": {"only": 1.0}, "obj": {"a": 0.5, "b": 0.5}})
    qs = {**QS, "one": choice("Only option?", ["only"]),
          "obj": choice("Structured?", {"a": {"means": "first"}, "b": ["second", "option"]})}
    out = cascade(t1, k5, thresholds=[0.5], protocols=["systemone", "jevk5"]).systemone({"state": STATE,
                                                                                          "questions": qs})
    sent = seen[0]["questions"]
    assert set(sent) == {"n", "c", "s", "obj"}                             # 'one' is kept by tier 1 (confidence 1)
    assert sent["s"]["criteria"] == ["Calm", "Frustrated: clearly unhappy", "Very angry"]
    assert sent["obj"]["criteria"] == {"a": '{"means":"first"}', "b": '["second","option"]'}
    assert seen[0]["model"] == "model-2"
    a = out["answers"]
    assert {k: v["tier"] for k, v in a.items()} == {"n": 2, "c": 2, "s": 2, "one": 1, "obj": 2}
    from jev_style.fake import jevk5_response
    raw = jevk5_response(seen[0])["answers"]                 # what the tier returned (deterministic)
    assert sent == jevk5_questions({k: v for k, v in qs.items() if k != "one"})
    assert a["n"]["noul"] == raw["n"]["noul"]
    p = list(raw["c"]["probabilities"].values())
    assert raw["c"]["confidence"] == max(p)                  # jevk5 reports p_max ...
    assert a["c"]["confidence"] == pytest.approx((3 * max(p) - 1) / 2)    # ... the cascade reports its own
    assert a["s"]["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}  # jevk5 sends no legend
    assert out["usage"]["input_tokens"] > 50


def test_jevk5_one_option_choice_is_not_sent_and_400_is_a_tier_error():
    seen = []
    k5 = JevStyle(base_url="http://k5", http_client=httpx.Client(base_url="http://k5", transport=jevk5_transport(seen)))
    t1 = Scripted(fail=ApiError(422, "input_budget_exceeded", "too long"))
    a = cascade(t1, k5, thresholds=[0.5], protocols=["systemone", "jevk5"])
    with pytest.raises(ApiError) as ei:                  # tier 1 failed, jevk5 cannot take it: tier 1's error
        a.systemone({"state": "s", "questions": {"one": choice("Only?", ["only"])}})
    assert ei.value.code == "input_budget_exceeded" and seen == []
    err = TierError.from_exception(JevStyleError(400, "http_error", "unknown question type 'x'",
                                                 body={"error": "unknown question type 'x'"}))
    assert (err.status, err.code) == (502, "tier_error")
    with pytest.raises(JevStyleError) as ei:            # the client reads {"error": "<string>"} bodies
        k5.decide("s", {"x": {"type": "choice", "instructions": "a", "criteria": {"only": None}}})
    assert ei.value.status == 400 and ei.value.message == "choice criteria must name at least two options"


def test_upstream_auth_failure_is_not_reported_as_the_callers():
    err = TierError.from_exception(JevStyleError(401, "unauthorized", "missing key", body={"error": {
        "code": "unauthorized", "message": "missing key"}}))
    assert (err.status, err.code) == (502, "tier_unauthorized")
    kept = TierError.from_exception(JevStyleError(422, "input_budget_exceeded", "long", body={"error": {
        "code": "input_budget_exceeded", "message": "long"}}))
    assert (kept.status, kept.code) == (422, "input_budget_exceeded")


# ----------------------------------------------------------------------------- config
def test_config_round_trip_and_file(tmp_path):
    cfg = {"$schema": "x", "id": "glue-2", "description": "2B -> JevK5-9B",
           "tiers": [{"name": "2b", "model_id": "jev-style-2b-decision-v3", "target": "local:2b-v3:mlx"},
                     {"name": "k5", "model_id": "alibiserikbay/JevK5-9B", "target": "http://127.0.0.1:8090",
                      "revision": "v0.3.3", "protocol": "jevk5", "max_options": 16, "timeout_s": 30}],
           "thresholds": [0.6], "confidence": "normalized_pmax", "frozen_utc": "2026-10-03T00:00:00Z",
           "calibration_sha256": "a" * 64}
    c = CascadeConfig.from_dict(cfg)
    assert c.tiers[1].protocol == "jevk5" and c.thresholds == (0.6,) and c.tiers[1].timeout_s == 30.0
    assert CascadeConfig.from_dict(c.to_dict()) == c
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    loaded = load_config(p)
    assert loaded == c and loaded.sha256 and len(loaded.sha256) == 64


@pytest.mark.parametrize("change,msg", [
    ({"thresholds": [0.5, 0.5]}, "thresholds must be an array of 1"),
    ({"thresholds": [1.5]}, "thresholds[0]"),
    ({"thresholds": [True]}, "thresholds[0]"),
    ({"threshold": [0.5]}, "unknown key"),
    ({"confidence": "pmax"}, "confidence must be"),
    ({"frozen_utc": "2026-10-03T00:00:00+02:00"}, "UTC"),
    ({"frozen_utc": "yesterday"}, "ISO 8601"),
    ({"calibration_sha256": "abc"}, "calibration_sha256"),
    ({"id": ""}, "id must be"),
    ({"tiers": [{"name": "a", "model_id": "m", "target": "fake"}]}, "at least two"),
    ({"tiers": [{"name": "a", "model_id": "m", "target": "fake"}, {"name": "a", "model_id": "m", "target": "fake"}]},
     "unique"),
    ({"tiers": [{"name": "a", "model_id": "m", "target": "ftp://x"}, {"name": "b", "model_id": "m",
                                                                        "target": "fake"}]}, "unknown engine target"),
    ({"tiers": [{"name": "a", "model_id": "m", "target": "cascade:x.json"}, {"name": "b", "model_id": "m",
                                                                               "target": "fake"}]}, "a tier must be"),
    ({"tiers": [{"name": "a", "model_id": "m", "target": "local:9b-v9"}, {"name": "b", "model_id": "m",
                                                                            "target": "fake"}]}, "unknown release"),
    ({"tiers": [{"name": "a", "model_id": "m", "target": "fake", "max_options": 0},
                {"name": "b", "model_id": "m", "target": "fake"}]}, "max_options"),
    ({"tiers": [{"name": "a", "model_id": "m", "target": "fake", "protocol": "grpc"},
                {"name": "b", "model_id": "m", "target": "fake"}]}, "protocol"),
    ({"tiers": [{"name": "a", "model_id": "m", "target": "fake", "api_key_env": "K"},
                {"name": "b", "model_id": "m", "target": "fake"}]}, "only apply to http"),
    ({"tiers": [{"name": "a", "target": "fake"}, {"name": "b", "model_id": "m", "target": "fake"}]}, "missing"),
])
def test_config_validation(change, msg):
    with pytest.raises(CascadeConfigError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        CascadeConfig.from_dict(_fake_config(**change))


def test_adapter_rejects_a_bad_threshold_count():
    with pytest.raises(ValueError):
        cascade(Scripted(), Scripted(), thresholds=[])
    with pytest.raises(ValueError):
        cascade(Scripted(), Scripted(), thresholds=[2.0])


def test_http_tier_api_key_env(monkeypatch):
    cfg = _fake_config(tiers=[{"name": "a", "model_id": "m", "target": "fake"},
                              {"name": "b", "model_id": "m", "target": "http://127.0.0.1:1", "api_key_env": "K5"}])
    monkeypatch.delenv("K5", raising=False)
    with pytest.raises(CascadeConfigError, match="K5"):
        CascadeAdapter.from_config(cfg)
    monkeypatch.setenv("K5", "sekret")
    a = CascadeAdapter.from_config(cfg)
    assert a.tiers[1].engine.api_key == "sekret" and a.tiers[1].backend == "http"


# ----------------------------------------------------------------------------- wire-up
def test_targets():
    assert parse_target("cascade:/x/c.json") == ("cascade", {"cascade": "/x/c.json"})
    assert parse_target("local:2b:mlx") == ("local", {"release": "2b", "backend": "mlx"})
    with pytest.raises(ValueError):
        parse_target("hf:")


def test_client_eval_serve_and_cli_take_a_cascade(tmp_path, capsys):
    p = tmp_path / "cascade.json"
    p.write_text(json.dumps(_fake_config()))
    js = JevStyle(cascade=str(p))
    out = js.decide(STATE, {"b": noul("About billing?")})
    assert out["model"] == "fake-cascade" and out["answers"]["b"]["tier"] in (1, 2)
    assert js.models()["data"][0]["config_sha256"]
    with pytest.raises(ValueError):
        JevStyle(cascade=str(p), fake=True)

    from jev_style.evaluate import build_engine, main as eval_main
    assert build_engine(f"cascade:{p}", "auto")(STATE, {"b": noul("x")})["backend"] == "cascade"
    ex = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / "support_tickets.jsonl")
    assert eval_main([ex, "--server", "fake", "--server", f"glue=cascade:{p}"]) == 0
    assert "glue" in capsys.readouterr().out

    from jev_style.server import build_app
    app = build_app(cascade=str(p))
    assert app.state.adapter.backend == "cascade"
    with pytest.raises(ValueError):
        build_app(cascade=str(p), fake=True)

    from jev_style.cli import main
    assert main(["decide", STATE, "--cascade", str(p), "--noul", "About billing?"]) == 0
    assert json.loads(capsys.readouterr().out)["answers"]["noul_1"]["tier_model"].startswith("jev-style-fake")
    assert main(["decide", STATE, "--cascade", str(p), "--fake", "--noul", "x"]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_fake_config(thresholds=[])))
    assert main(["decide", STATE, "--cascade", str(bad), "--noul", "x"]) == 2
    assert "thresholds" in capsys.readouterr().err


def test_examples_are_valid():
    from pathlib import Path
    for f in sorted((Path(__file__).resolve().parents[1] / "examples" / "cascade").glob("*.json")):
        c = load_config(f)
        assert c.tiers[-1].protocol == "jevk5" and len(c.thresholds) == len(c.tiers) - 1
        assert not math.isnan(sum(c.thresholds))
