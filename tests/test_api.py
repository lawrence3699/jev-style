"""The HTTP contract, on the fake engine (no model needed)."""
import pytest
from fastapi.testclient import TestClient

from jev_style.server import build_app

TICKET = {
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"returns": "exchanges", "shipping": "delays", "billing": "charges"}},
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", {"label": "Frustrated", "description": "clearly unhappy"}, "Very angry"]},
    },
}


def test_healthz(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["model"] == "jev-style-fake"
    assert "agent_approval" in h["extensions"] and h["ext_errors"] == []


def test_systemone_shape(client):
    r = client.post("/v1/systemone", json=TICKET)
    assert r.status_code == 200 and r.headers["x-request-id"].startswith("req_")
    a = r.json()["answers"]
    assert set(a) == {"department", "escalate", "frustration"}
    ch = a["department"]
    assert ch["choice"] in ("returns", "shipping", "billing")
    assert list(ch["probabilities"]) == ["returns", "shipping", "billing"]
    assert abs(sum(ch["probabilities"].values()) - 1) < 1e-9 and 0 <= ch["confidence"] <= 1
    assert 0 <= a["escalate"]["noul"] <= 1
    sc = a["frustration"]
    assert sc["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert 0 <= sc["score"] <= 2
    body = r.json()
    assert body["usage"]["output_tokens"] == 0 and body["usage"]["input_tokens"] > 0
    assert "total_ms" in body["timing"]


def test_deterministic(client):
    a = client.post("/v1/systemone", json=TICKET).json()["answers"]
    b = client.post("/v1/systemone", json=TICKET).json()["answers"]
    assert a == b


def test_structured_state(client):
    r = client.post("/v1/systemone", json={"state": {"ticket": "hi", "tier": "pro"},
                                           "questions": {"x": {"type": "noul", "instructions": "ok?"}}})
    assert r.status_code == 200


@pytest.mark.parametrize("body,code", [
    ({"questions": {"x": {"type": "noul", "instructions": "a"}}}, "invalid_request"),
    ({"state": "s", "questions": {}}, "invalid_request"),
    ({"state": "s", "questions": {"x": {"type": "maybe", "instructions": "a"}}}, "invalid_question"),
    ({"state": "s", "questions": {"x": {"type": "score", "instructions": "a", "criteria": ["one"]}}},
     "invalid_question"),
    ({"state": "s", "questions": {"x": {"type": "choice", "instructions": "a"}}}, "invalid_question"),
])
def test_validation(client, body, code):
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


def test_invalid_json(client):
    r = client.post("/v1/systemone", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_json"


def test_budget_never_truncates(client):
    r = client.post("/v1/systemone", json={"state": "word " * 30_000,
                                           "questions": {"x": {"type": "noul", "instructions": "long?"}}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "max_tokens_exceeded"


def test_models(client):
    m = client.get("/v1/models").json()
    assert m["data"][0]["context_tokens"] == 25_600 and m["models"][0]["name"] == "jev-style-fake"


def test_auth():
    c = TestClient(build_app(fake=True, api_key="sekret"), base_url="http://127.0.0.1")
    assert c.post("/v1/systemone", json=TICKET).status_code == 401
    ok = c.post("/v1/systemone", json=TICKET, headers={"authorization": "Bearer sekret"})
    assert ok.status_code == 200
    assert c.get("/healthz").status_code == 200


def test_web_and_demos(client):
    assert client.get("/").status_code == 200
    demos = client.get("/api/demos").json()
    assert {d["slug"] for d in demos["demos"]} == {"agent-approval", "snake", "chinese"} and not demos["errors"]
    assert client.get("/demo-data/chinese/manifest.json").status_code == 200
    assert client.get("/nope.html").status_code == 404


def test_agent_approval_demo(client):
    feed = client.get("/api/agent-approval/feed").json()
    assert feed["counts"]["items"] > 0
    r = client.post("/api/agent-approval/check", json={"id": feed["items"][0]["id"]}).json()
    assert r["decision"] in ("allow", "ask", "deny")


# ----------------------------------------------------------------------------- cross-site requests are refused
def test_host_and_origin_checks_keep_other_sites_out(client):
    r = client.post("/v1/systemone", json=TICKET, headers={"host": "attacker.example"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden_host"
    r = client.post("/v1/systemone", content=b'{"state": "s", "questions": {"x": {"type": "noul", "instructions": "a"}}}',
                    headers={"content-type": "text/plain", "origin": "https://evil.example"})   # no preflight
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden_origin"
    assert client.get("/healthz", headers={"origin": "null"}).status_code == 403
    assert client.get("/", headers={"host": "localhost:8765"}).status_code == 200
    assert client.get("/healthz", headers={"host": "[::1]:8765"}).status_code == 200
    assert client.post("/v1/systemone", json=TICKET, headers={"origin": "http://127.0.0.1:8765"}).status_code == 200
    assert client.post("/v1/systemone", json=TICKET, headers={"origin": "http://localhost:3000"}).status_code == 200


def test_allowed_hosts_can_be_widened_or_switched_off():
    c = TestClient(build_app(fake=True, allowed_hosts=None), base_url="http://127.0.0.1")
    assert c.get("/healthz", headers={"host": "attacker.example", "origin": "https://evil.example"}).status_code == 200
    c = TestClient(build_app(fake=True, allowed_hosts={"box.lan"}), base_url="http://127.0.0.1")
    assert c.get("/healthz", headers={"host": "box.lan:8765", "origin": "http://box.lan:8765"}).status_code == 200
    assert c.get("/healthz", headers={"host": "box.lan:8765", "origin": "https://evil.example"}).status_code == 403
    assert c.get("/healthz", headers={"host": "other.lan"}).status_code == 403
    assert c.get("/healthz").status_code == 200                                      # loopback always
    c = TestClient(build_app(fake=True, allowed_hosts="bound"), base_url="http://127.0.0.1")   # --host 0.0.0.0
    assert c.get("/healthz", headers={"host": "10.0.0.5:8765"}).status_code == 200
    assert c.get("/healthz", headers={"host": "10.0.0.5:8765", "origin": "http://10.0.0.5:8765"}).status_code == 200
    assert c.get("/healthz", headers={"host": "10.0.0.5:8765", "origin": "https://evil.example"}).status_code == 403


def test_allowed_hosts_for_the_cli_bind_address():
    from jev_style.cli import allowed_hosts_for
    assert allowed_hosts_for("127.0.0.1") == {"127.0.0.1", "localhost", "::1"}
    assert allowed_hosts_for("box.lan") == {"127.0.0.1", "localhost", "::1", "box.lan"}
    assert allowed_hosts_for("0.0.0.0") == "bound" and allowed_hosts_for("::") == "bound"
