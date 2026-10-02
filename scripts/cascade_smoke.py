#!/usr/bin/env python3
"""Smoke test for the confidence cascade: two fake tiers, no model, no network beyond localhost.

    python scripts/cascade_smoke.py [--threshold 0.6]

Tier 1 is the in-process fake engine (``fake`` target). Tier 2 is a stand-in for ``jevk5-serve``: a stdlib HTTP
server on a free localhost port that answers in JevK5's shapes (``jev_style.fake.jevk5_response``; probabilities
from a different hash, so the two tiers disagree). The cascade runs in-process behind the real FastAPI app and
one request is printed: each answer carries ``tier`` / ``tier_model``, ``timing.tiers`` shows the per-tier calls.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # this checkout, not an installed jev-style

from fastapi.testclient import TestClient  # noqa: E402

from jev_style.cascade import CascadeAdapter  # noqa: E402
from jev_style.fake import jevk5_response  # noqa: E402
from jev_style.server import create_app  # noqa: E402


class FakeJevK5(BaseHTTPRequestHandler):
    """POST /v1/systemone like jevk5-serve 0.3.3: 200 with its answer shapes, 400 {"error": "<message>"}."""

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        try:
            status, payload = 200, jevk5_response(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        except (ValueError, KeyError, TypeError) as e:
            status, payload = 400, {"error": str(e)}
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, default=0.6, help="tier 1 keeps answers with confidence >= this")
    args = ap.parse_args()

    k5 = ThreadingHTTPServer(("127.0.0.1", 0), FakeJevK5)
    threading.Thread(target=k5.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{k5.server_address[1]}"
    config = {"id": "smoke-cascade", "description": "fake engine -> fake jevk5-serve",
              "tiers": [{"name": "small", "model_id": "jev-style-fake", "target": "fake"},
                        {"name": "big", "model_id": "fake-jevk5", "target": url, "protocol": "jevk5"}],
              "thresholds": [args.threshold]}
    client = TestClient(create_app(CascadeAdapter.from_config(config)))
    body = {"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
            "questions": {
                "department": {"type": "choice", "instructions": "Which team should handle this?",
                               "criteria": {"returns": "exchanges", "shipping": "delays", "billing": "charges"}},
                "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
                "refund": {"type": "noul", "instructions": "The customer asks for money back."},
                "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                                "criteria": ["Calm", {"label": "Frustrated", "description": "clearly unhappy"},
                                             "Very angry"]}}}
    r = client.post("/v1/systemone", json=body)
    print(f"tier 2 (fake jevk5-serve) at {url}; threshold {args.threshold}; HTTP {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    k5.shutdown()
    return 0 if r.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
