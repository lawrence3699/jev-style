"""Route through a running JevStyle server with Semantic Router 1.1.1.

Install in a Python 3.10–3.13 virtual environment:
    python -m pip install "semantic-router==1.1.1" "jev-style[mlx]"
Use jev-style[torch] instead of [mlx] outside Apple silicon.

Terminal 1:
    jev-style serve --release 0.8b
Terminal 2:
    python examples/semantic_router_local.py "Where is my parcel?"

No external API key or embedding service is needed. An optional JEV_STYLE_API_KEY
is forwarded only to the configured JevStyle endpoint. The default threshold is
illustrative: select and independently test it on your own labeled requests.
This script prints a proposed route; it does not execute a business action.
"""
from __future__ import annotations

import argparse
import json
import os

from jev_style import JevStyle
from semantic_router import DecisionRouter, Route
from semantic_router.decision_models import SystemOneDecisionModel


def route_query(query: str, base_url: str, threshold: float) -> dict:
    """Use the server's advertised model ID and return a route or review step."""
    api_key = os.environ.get("JEV_STYLE_API_KEY")
    with JevStyle(base_url=base_url, api_key=api_key, timeout=10.0) as client:
        model_id = client.models()["data"][0]["id"]

    model = SystemOneDecisionModel(
        name=model_id,
        base_url=base_url,
        api_key=api_key,
        timeout=30.0,
        retries=0,
    )
    router = DecisionRouter(
        model=model,
        routes=[
            Route(name="billing", description="Charges, invoices, and refunds"),
            Route(name="shipping", description="Parcels, tracking, and delivery"),
            Route(name="account", description="Login, passwords, and account settings"),
            Route(name="other", description="Requests outside the listed workflows"),
        ],
        confidence_threshold=threshold,
    )
    try:
        result = router(query)
        return {
            "model": model_id,
            "route": result.name,
            "probability": result.confidence,
            "next_step": (
                "human_or_llm_review" if result.name in (None, "other") else result.name
            ),
        }
    finally:
        router.close()


def probability(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="Where is my parcel?")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--threshold", type=probability, default=0.90,
        help="minimum probability; 0.90 is illustrative, not a validated setting",
    )
    args = parser.parse_args(argv)
    try:
        result = route_query(args.query, args.url, args.threshold)
    except Exception as exc:
        # A failed local call must not turn into an unreviewed business action.
        print(json.dumps({
            "route": None,
            "next_step": "human_or_llm_review",
            "error": type(exc).__name__,
            "hint": "Check that jev-style serve is running and --url is correct.",
        }, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
