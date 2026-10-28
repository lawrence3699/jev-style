"""Print routing recommendations; never updates a ticket system."""
from __future__ import annotations

import argparse
from time import perf_counter

import httpx
from jev_style import JevStyleError

try:
    from .common import QUESTIONS, add_arguments, client, emit, pick, read_tickets, summarize
except ImportError:
    from common import QUESTIONS, add_arguments, client, emit, pick, read_tickets, summarize


def route(js, ticket, minimum=0.8):
    started = perf_counter()
    result = {"id": ticket["id"], "expected_team": ticket.get("expected_team"), "decision": "human_review"}
    try:
        response = js.decide(ticket["text"], QUESTIONS)
        team, probability = pick(response)
        result.update(model=response.get("model"), suggested_team=team, top_probability=probability)
        if probability >= minimum:
            result["decision"] = team
    except (JevStyleError, httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        result["error"] = type(error).__name__
    result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    rows = read_tickets(args.input)
    with client(args.url, args.fake) as js:
        results = [route(js, row, args.min_probability) for row in rows]
    for result in results:
        emit(result)
    emit(summarize(results, args.fake))


if __name__ == "__main__":
    main()
