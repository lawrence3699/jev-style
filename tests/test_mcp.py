import asyncio

import pytest

pytest.importorskip("mcp")

from jev_style.mcp_server import BackendError, HttpBackend, InProcessBackend, build_instructions, build_server  # noqa: E402


def test_tools_listed_and_callable():
    srv = build_server(InProcessBackend(fake=True))
    tools = {t.name for t in asyncio.run(srv.list_tools())}
    assert {"decide", "noul", "choice", "score", "model_info"} <= tools
    res = asyncio.run(srv.call_tool("choice", {"state": "charged twice", "question": "Team?",
                                               "options": ["billing", "tech"]}))
    assert not getattr(res, "isError", False)


# ----------------------------------------------------------------------------- HTTP errors, both body shapes
@pytest.mark.parametrize("status,body,text", [
    (401, {"error": "unauthorized", "message": "Missing or invalid API key."},
     "401 unauthorized: Missing or invalid API key."),                                       # hosted API: flat
    (413, {"error": "input_too_large", "message": "Input exceeds the rendered token budget."},
     "413 input_too_large: Input exceeds the rendered token budget."),
    (400, {"error": {"code": "max_tokens_exceeded", "message": "too long"}}, "400 max_tokens_exceeded: too long"),
    (422, {"error": {"code": "invalid_question", "message": "bad", "question": "q"}},
     "422 invalid_question: bad (question 'q')"),
    (400, {"error": "unknown question type 'x'"}, "400 http_error: unknown question type 'x'"),   # jevk5: bare
    (502, b"<html>bad gateway</html>", "502 http_error: Bad Gateway"),
])
def test_http_backend_reports_the_servers_error(stub_server, status, body, text):
    srv = stub_server(status, body)
    with pytest.raises(BackendError) as ei:
        HttpBackend(srv.url).decide({"state": "s", "questions": {"q": {"type": "noul", "instructions": "a"}}})
    assert str(ei.value) == text


def test_http_backend_error_reaches_the_tool_as_a_tool_error(stub_server):
    srv = stub_server(401, {"error": "unauthorized", "message": "Invalid or revoked API key."})
    mcp = build_server(HttpBackend(srv.url))
    with pytest.raises(Exception) as ei:                 # the library's ToolError, with the server's reason in it
        asyncio.run(mcp.call_tool("noul", {"state": "s", "question": "Is it?"}))
    assert "401 unauthorized: Invalid or revoked API key." in str(ei.value)
    assert "AttributeError" not in str(ei.value)


# ----------------------------------------------------------------------------- instructions follow the backend
def test_instructions_in_process_carry_the_fake_engines_budgets():
    text = build_server(InProcessBackend(fake=True)).instructions
    assert "up to 25,600 tokens" in text and "must fit in 2,048 tokens" in text
    assert "inside this process; nothing is sent elsewhere" in text


def test_instructions_for_a_local_server_read_its_limits(stub_server):
    srv = stub_server(200, {"data": [{"id": "jev-style-2b-decision-v4", "context_tokens": 32768,
                                      "head_max_tokens": 32768}]})
    text = build_instructions(HttpBackend(srv.url))
    assert "up to 32,768 tokens" in text and "must fit in" not in text
    assert f"local server at {srv.url}" in text and "nothing is sent elsewhere" in text


def test_instructions_for_a_remote_server_say_the_state_leaves_this_machine():
    with pytest.raises(BackendError, match="--allow-remote"):
        HttpBackend("https://api.example.com")
    text = build_instructions(HttpBackend("https://api.example.com", allow_remote=True))   # not reachable: no numbers
    assert "Requests go to the remote server https://api.example.com" in text
    assert "nothing is sent elsewhere" not in text and "up to the model's context length" in text


def test_main_refuses_an_unknown_env_release(monkeypatch, capsys):
    from jev_style.mcp_server import main
    monkeypatch.setenv("JEV_STYLE_RELEASE", "9b")
    assert main(["--fake"]) == 2
    assert "JEV_STYLE_RELEASE='9b'" in capsys.readouterr().err
