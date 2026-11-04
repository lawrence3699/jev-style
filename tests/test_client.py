import pytest
from fastapi.testclient import TestClient

from jev_style import JevStyle, JevStyleError, choice, noul, score
from jev_style.server import build_app


def _qs():
    return {"b": noul("About billing?"), "t": choice("Team?", ["billing", "tech"]),
            "u": score("Urgency?", ["low", "high"])}


def test_in_process_matches_http():
    local = JevStyle(fake=True).decide("I was charged twice.", _qs())
    http = JevStyle(base_url="http://test", http_client=TestClient(build_app(fake=True), base_url="http://127.0.0.1")).decide(
        "I was charged twice.", _qs())
    assert local["answers"] == http["answers"]


def test_errors_are_typed():
    js = JevStyle(fake=True)
    try:
        js.decide("x", {"q": {"type": "choice", "instructions": "a", "criteria": {}}})
    except JevStyleError as e:
        assert e.status == 422 and e.question == "q"
    else:
        raise AssertionError("expected JevStyleError")


# ----------------------------------------------------------------------------- error bodies, both shapes
def test_parse_error_body_reads_both_shapes():
    from jev_style.client import parse_error_body
    flat = {"error": "unauthorized", "message": "Missing or invalid API key."}
    assert parse_error_body(flat) == {"code": "unauthorized", "message": "Missing or invalid API key."}
    nested = {"error": {"code": "invalid_question", "message": "bad", "question": "q"}}
    assert parse_error_body(nested) == {"code": "invalid_question", "message": "bad", "question": "q"}
    assert parse_error_body({"error": "a bare message"}) == {"code": "http_error", "message": "a bare message"}
    assert parse_error_body({"error": "x", "message": "m", "limit": "input_tokens"})["code"] == "x"
    assert parse_error_body("nope", 401) == {"code": "unauthorized", "message": ""}
    assert parse_error_body({}, 500) == {"code": "http_error", "message": ""}


@pytest.mark.parametrize("status,body,code,too_long", [
    (401, {"error": "unauthorized", "message": "Missing or invalid API key."}, "unauthorized", False),
    (413, {"error": "input_too_large", "message": "Input exceeds the rendered token budget."}, "input_too_large", True),
    (400, {"error": {"code": "max_tokens_exceeded", "message": "input needs 30000 tokens"}}, "max_tokens_exceeded",
     True),
    (422, {"error": {"code": "input_budget_exceeded", "message": "old server"}}, "input_budget_exceeded", True),
])
def test_client_keeps_the_hosted_apis_error_code(status, body, code, too_long):
    import httpx
    http = httpx.Client(base_url="https://api.example", transport=httpx.MockTransport(
        lambda req: httpx.Response(status, json=body)))
    js = JevStyle(base_url="https://api.example", http_client=http, api_key="k")
    with pytest.raises(JevStyleError) as ei:
        js.decide("s", {"q": noul("a")})
    e = ei.value
    assert (e.status, e.code, e.message) == (status, code, body["message"] if "message" in body
                                             else body["error"]["message"])
    assert e.input_too_long is too_long and e.body == body


# ----------------------------------------------------------------------------- where $JEV_STYLE_API_KEY may go
def test_env_key_goes_to_loopback_or_the_configured_server_only(monkeypatch):
    monkeypatch.setenv("JEV_STYLE_API_KEY", "local-secret")
    monkeypatch.delenv("JEV_STYLE_URL", raising=False)
    assert JevStyle(base_url="http://127.0.0.1:8765").api_key == "local-secret"
    assert JevStyle(base_url="http://localhost:8766/").api_key == "local-secret"
    assert JevStyle(base_url="https://other-vendor.example/v1").api_key is None       # never by accident
    assert JevStyle(base_url="https://other-vendor.example", api_key="theirs").api_key == "theirs"
    assert JevStyle(fake=True).api_key == "local-secret"                               # in-process: never sent
    monkeypatch.setenv("JEV_STYLE_URL", "https://api.jevstyle.com/")
    assert JevStyle(base_url="https://api.jevstyle.com").api_key == "local-secret"     # the configured server
    assert JevStyle(base_url="https://api.jevstyle.com/v1").api_key is None
    from jev_style.client import from_target
    assert from_target("https://third.example").api_key is None
    assert from_target("http://127.0.0.1:9000").api_key == "local-secret"
