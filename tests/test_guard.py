"""The guard's failure semantics: errors, odd answers and thresholds never turn into a silent allow."""
import json

import pytest

from jev_style import guard
from jev_style.client import JevStyle
from jev_style.guard_replay import main as replay

CALL = {"tool_name": "Bash", "tool_input": {"command": "ls -la"}, "cwd": "/home/u/proj", "project_dir": "/home/u/proj",
        "home": "/home/u"}


def _caller():
    js = JevStyle(fake=True)
    return lambda body: js.decide(body["state"], body["questions"])


def _answers(destructive=0.1, exfiltration=0.1, outside_project=0.1, secrets=0.1, risk=0.5):
    """A well-formed response with chosen values (noul = P(true), risk = expected level index)."""
    return {"model": "jev-style-fake", "answers": {
        "destructive": {"type": "noul", "noul": destructive},
        "exfiltration": {"type": "noul", "noul": exfiltration},
        "outside_project": {"type": "noul", "noul": outside_project},
        "secrets": {"type": "noul", "noul": secrets},
        "risk": {"type": "score", "score": risk, "legend": {}, "probabilities": {}, "confidence": 0.5}},
        "usage": {"input_tokens": 100}, "timing": {"total_ms": 1.0}}


def _cfg(**over):
    cfg = guard.load_config()
    cfg.update(over)
    return cfg


def test_hard_rules_and_hook_output():
    cfg = guard.load_config()
    res = guard.evaluate({"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}, "cwd": "/work/p"}, cfg,
                         call=_caller())
    assert res["decision"] in ("ask", "deny")


def test_server_down_asks(monkeypatch):
    cfg = guard.load_config()
    cfg["server_url"] = "http://127.0.0.1:9"          # nothing listens here
    cfg["timeout_s"] = 1
    out = guard.run_hook(json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}, "cwd": "/tmp"}), cfg)
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_replay_fake(capsys):
    assert replay(["--fake"]) == 0
    assert "agreement" in capsys.readouterr().out


# ----------------------------------------------------------------------------- thresholds
@pytest.mark.parametrize("values,expected", [
    ({}, "allow"),
    ({"destructive": 0.6}, "ask"),                   # ask at 0.5
    ({"destructive": 0.95}, "deny"),                 # deny at 0.93
    ({"secrets": 0.96}, "deny"),
    ({"outside_project": 0.7}, "ask"),               # outside_project has no deny bound
    ({"outside_project": 0.99}, "ask"),
    ({"risk": 2.5}, "ask"),                          # risk: ask at 2.0, deny at 3.4
    ({"risk": 3.5}, "deny"),
])
def test_thresholds_turn_values_into_verdicts(values, expected):
    res = guard.evaluate(CALL, _cfg(), call=lambda body: _answers(**values), use_rules=False)
    assert (res["decision"], res["source"]) == (expected, "model")
    assert set(res["values"]) == {"destructive", "exfiltration", "outside_project", "secrets", "risk"}


def test_emit_allow_prints_allow_only_for_a_real_verdict():
    cfg = _cfg(emit_allow=True)
    res = guard.evaluate(CALL, cfg, call=lambda body: _answers(), use_rules=False)
    out = guard.hook_output(res, cfg)
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert guard.hook_output(res, _cfg(emit_allow=False)) is None          # default: leave it to Claude Code


# ----------------------------------------------------------------------------- 200 without usable answers
@pytest.mark.parametrize("resp", [
    {"answers": {}},
    {"model": "x"},
    {"answers": {"destructive": {"type": "noul", "noul": 0.0}}},                       # four questions missing
    {"answers": {q: {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}, "confidence": 1.0}
                 for q in ("destructive", "exfiltration", "outside_project", "secrets", "risk")}},   # wrong type
    {"answers": {q: {"type": "noul"} for q in ("destructive", "exfiltration", "outside_project", "secrets", "risk")}},
    [],
    "nope",
])
def test_200_without_usable_answers_is_ask_even_with_emit_allow(resp):
    cfg = _cfg(emit_allow=True)
    res = guard.evaluate(CALL, cfg, call=lambda body: resp, use_rules=False)
    assert (res["decision"], res["source"]) == ("ask", "error")
    assert "usable value" in res["error"] or "JSON object" in res["error"]
    out = guard.hook_output(res, cfg)["hookSpecificOutput"]
    assert out["permissionDecision"] == "ask" and "could not check" in out["permissionDecisionReason"]


def test_apply_thresholds_reports_missing_questions():
    v = guard.apply_thresholds({"destructive": {"type": "noul", "noul": 0.2}}, guard.DEFAULTS["thresholds"],
                               expected_types={"destructive": "noul", "risk": "score"})
    assert v["decision"] == "allow" and v["values"] == {"destructive": 0.2}
    assert v["missing"] == ["exfiltration", "outside_project", "secrets", "risk"]
    v = guard.apply_thresholds({"destructive": {"type": "score", "score": 0.2}}, {"destructive": {"ask": 0.5}},
                               expected_types={"destructive": "noul"})
    assert v["missing"] == ["destructive"]


# ----------------------------------------------------------------------------- HTTP errors, both body shapes
@pytest.mark.parametrize("status,body,code,message", [
    (401, {"error": "unauthorized", "message": "Missing or invalid API key."}, "unauthorized",
     "Missing or invalid API key."),                                                         # hosted API: flat
    (413, {"error": "input_too_large", "message": "Input exceeds the rendered token budget.", "limit": "input_tokens"},
     "input_too_large", "Input exceeds the rendered token budget."),
    (400, {"error": {"code": "max_tokens_exceeded", "message": "input needs 30000 tokens"}}, "max_tokens_exceeded",
     "input needs 30000 tokens"),                                                              # this package: nested
    (422, {"error": {"code": "invalid_question", "message": "bad", "question": "risk"}}, "invalid_question", "bad"),
    (400, {"error": "choice criteria must name at least two options"}, "",
     "choice criteria must name at least two options"),                                        # jevk5: bare message
    (500, b"<html>oops</html>", "", ""),
])
def test_http_error_bodies_become_ask_with_the_servers_reason(stub_server, status, body, code, message):
    srv = stub_server(status, body)
    cfg = _cfg(server_url=srv.url, timeout_s=5, emit_allow=True)
    res = guard.evaluate(CALL, cfg, use_rules=False)
    assert (res["decision"], res["source"]) == ("ask", "error")
    assert res["error"].startswith(f"server returned {status}")
    assert code in res["error"] and message in res["error"]
    assert "AttributeError" not in res["error"] and "object has no attribute" not in res["error"]
    assert srv.requests[0]["path"] == "/v1/systemone"


def test_http_200_goes_through_the_thresholds(stub_server):
    srv = stub_server(200, _answers(secrets=0.99))
    res = guard.evaluate(CALL, _cfg(server_url=srv.url, timeout_s=5), use_rules=False)
    assert (res["decision"], res["source"]) == ("deny", "model")
    sent = json.loads(srv.requests[0]["body"])
    assert set(sent["questions"]) == {"destructive", "exfiltration", "outside_project", "secrets", "risk"}
    assert sent["state"]["tool"] == "Bash"


def test_error_fields_reads_both_shapes():
    assert guard.error_fields({"error": "unauthorized", "message": "m"}) == {"code": "unauthorized", "message": "m"}
    assert guard.error_fields({"error": {"code": "c", "message": "m"}}) == {"code": "c", "message": "m"}
    assert guard.error_fields({"error": "just a message"}) == {"code": "", "message": "just a message"}
    assert guard.error_fields({"message": "m"}) == {"code": "", "message": "m"}
    assert guard.error_fields([1, 2]) == {"code": "", "message": ""}
    assert guard.error_fields({"error": 5}) == {"code": "", "message": ""}


def test_help_names_the_subcommand(capsys):
    with pytest.raises(SystemExit):
        guard.main(["--help"])
    assert capsys.readouterr().out.startswith("usage: jev-style guard ")
    with pytest.raises(SystemExit):
        replay(["--help"])
    assert capsys.readouterr().out.startswith("usage: jev-style guard-replay ")
