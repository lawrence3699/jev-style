from fastapi.testclient import TestClient

from jev_style import JevStyle, JevStyleError, choice, noul, score
from jev_style.server import build_app


def _qs():
    return {"b": noul("About billing?"), "t": choice("Team?", ["billing", "tech"]),
            "u": score("Urgency?", ["low", "high"])}


def test_in_process_matches_http():
    local = JevStyle(fake=True).decide("I was charged twice.", _qs())
    http = JevStyle(base_url="http://test", http_client=TestClient(build_app(fake=True))).decide(
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
