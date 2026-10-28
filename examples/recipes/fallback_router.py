"""Ask a second systemone model only for uncertain or failed primary requests."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from time import perf_counter

import httpx
from jev_style import JevStyleError

try:
    from .common import QUESTIONS, add_arguments, client, emit, pick, read_tickets, summarize
except ImportError:
    from common import QUESTIONS, add_arguments, client, emit, pick, read_tickets, summarize


def route(primary, fallback, ticket, minimum=0.8):
    started = perf_counter()
    result = {"id": ticket["id"], "expected_team": ticket.get("expected_team"),
              "decision": "human_review", "source": "human_review", "fallback_called": False}
    for name, js in (("primary", primary), ("fallback", fallback)):
        if js is None:
            continue
        if name == "fallback":
            result["fallback_called"] = True
        try:
            response = js.decide(ticket["text"], QUESTIONS)
            team, probability = pick(response)
            result[name] = {"model": response.get("model"), "suggested_team": team,
                            "top_probability": probability}
            if probability >= minimum:
                result.update(decision=team, source=name)
                break
        except (JevStyleError, httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            result[name] = {"error": type(error).__name__}
    result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--fallback-url", help="explicit URL of the second systemone-compatible server")
    args = parser.parse_args()
    rows = read_tickets(args.input)
    with ExitStack() as stack:
        primary = stack.enter_context(client(args.url, args.fake))
        fallback = (stack.enter_context(client(args.fallback_url, args.fake))
                    if args.fallback_url or args.fake else None)
        results = [route(primary, fallback, row, args.min_probability) for row in rows]
    for result in results:
        emit(result)
    summary = summarize(results, args.fake)
    summary.update(primary_accepted=sum(r["source"] == "primary" for r in results),
                   fallback_calls=sum(r["fallback_called"] for r in results),
                   fallback_accepted=sum(r["source"] == "fallback" for r in results))
    emit(summary)


if __name__ == "__main__":
    main()
