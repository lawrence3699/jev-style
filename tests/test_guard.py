import json

from jev_style import guard
from jev_style.client import JevStyle
from jev_style.guard_replay import main as replay


def _caller():
    js = JevStyle(fake=True)
    return lambda body: js.decide(body["state"], body["questions"])


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
