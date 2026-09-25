#!/usr/bin/env python3
"""Find LLM calls in a codebase that only return a decision (a label, yes/no or a rating).

Those calls are candidates for a small local decision model: they send a prompt to a hosted LLM and
then parse one word, a boolean or a number out of the reply. Standard library only.

    python3 find_decisions.py [PATH ...] [--json out.json] [--min-score 3]

Heuristic, not a parser: every hit is a lead for a person (or an agent) to read. For each LLM call site
it looks at a window of lines around the call and scores decision-shaped signals:

* prompt wording: classify, categorize, label, "yes or no", "true or false", "respond with only",
  "one of", "rate ... 1 to 5", sentiment, intent, route, is this / does this, should ...;
* structured tiny outputs: enum / Literal / bool fields in a response schema, ``max_tokens`` <= 10,
  ``temperature=0`` with a short answer, ``.strip().lower() in (...)``, ``== "yes"``, ``int(reply)``;

and guesses the question type: noul (yes/no), choice (labels) or score (ratings).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rb", ".java", ".kt", ".php", ".rs", ".cs"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "dist", "build", "__pycache__", ".next", "vendor",
             "target", ".tox", ".mypy_cache", "site-packages", "coverage"}
MAX_BYTES = 1_000_000
WINDOW = 40                       # at most this many lines after the call site (reply parsing lives here)
BEFORE = 25                       # at most this many lines before it (the prompt is often built just above)
TOP = re.compile(r"^(def |async def |class |function |async function |export |const |let |var |func |fn |pub fn |"
                 r"public |private |@)")                     # a new top-level definition ends the window
CONST = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")              # module-level prompt constants used in the call

CALL = re.compile(
    r"(chat\.completions\.create|completions\.create|responses\.create|messages\.create|\.messages\.stream|"
    r"generate_content|GenerativeModel|litellm\.(a?completion)|ChatOpenAI|ChatAnthropic|ChatGoogleGenerativeAI|"
    r"\binvoke\(|\bainvoke\(|ollama\.(chat|generate)|generateText|generateObject|streamText|"
    r"openai\.ChatCompletion|anthropic\.Anthropic|client\.chat\(|/v1/chat/completions|/v1/messages|"
    r"structured_output|with_structured_output|instructor\.)", re.I)

SIGNALS: list[tuple[str, re.Pattern, int, str | None]] = [
    ("classify", re.compile(r"\bclassif(y|ies|ication)|categori[sz]e|\bcategory\b|\blabel(s|ed)?\b", re.I), 2, "choice"),
    ("yes/no", re.compile(r"yes\s*(or|/)\s*no|true\s*(or|/)\s*false|\byes\b.*\bno\b", re.I), 3, "noul"),
    ("only-answer", re.compile(r"(respond|answer|reply|return|output)\s+(with\s+)?(only|just|a single|one word|exactly)",
                               re.I), 2, None),
    ("one-of", re.compile(r"\bone of\b|\bchoose (from|between|one)|\bpick (one|the best)|\bselect (one|the)", re.I), 2,
     "choice"),
    ("rating", re.compile(r"\brate\b|\brating\b|\bscore\b.*\b(1|0)\s*(-|to)\s*(5|10|100)\b|\bon a scale\b|likert",
                          re.I), 3, "score"),
    ("sentiment", re.compile(r"sentiment|toxic|spam|nsfw|offensive|harmful|jailbreak|moderat", re.I), 2, "choice"),
    ("intent/route", re.compile(r"\bintent\b|\broute\b|\brouting\b|\bdepartment\b|\bwhich (team|agent|tool)", re.I), 2,
     "choice"),
    ("is-this", re.compile(r"\b(is|does|should|can|has) (this|the|it)\b[^.?\n]{0,80}\?", re.I), 1, "noul"),
    ("enum-schema", re.compile(r"\bLiteral\[|\benum\b|\bEnum\)|z\.enum\(|\"enum\"\s*:", re.I), 2, "choice"),
    ("bool-schema", re.compile(r":\s*bool\b|z\.boolean\(|\"type\"\s*:\s*\"boolean\"", re.I), 2, "noul"),
    ("tiny-output", re.compile(r"max_(completion_|output_)?tokens\s*[=:]\s*([1-9]|10)\b", re.I), 3, None),
    ("parse-label", re.compile(r"\.strip\(\)\.lower\(\)|\.lower\(\)\s*(==|in)|==\s*['\"](yes|no|true|false)['\"]|"
                               r"startswith\(['\"](yes|no)|\bint\((reply|response|answer|result|text|content)|"
                               r"parseInt\(|toLowerCase\(\)\s*===?", re.I), 2, None),
]


def iter_files(paths: list[str]):
    for root in paths:
        p = Path(root)
        if p.is_file():
            yield p
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for f in filenames:
                fp = Path(dirpath) / f
                if fp.suffix in EXTS:
                    yield fp


def call_window(lines: list[str], i: int) -> list[str]:
    """The call's enclosing top-level block (bounded), plus the definitions of CONSTANTS it uses."""
    lo = i
    while lo > max(0, i - BEFORE) and not TOP.match(lines[lo]):
        lo -= 1
    hi = i + 1
    while hi < min(len(lines), i + WINDOW) and not TOP.match(lines[hi]) and not CALL.search(lines[hi]):
        hi += 1
    block = lines[lo:hi]
    extra = []
    for name in sorted(set(CONST.findall("\n".join(block)))):
        for j, line in enumerate(lines):
            if re.match(rf"^\s*(export\s+)?(const\s+|let\s+|var\s+)?{name}\s*[:=]", line):
                extra.extend(lines[j:j + 12])
                break
    return extra + block


def scan_file(fp: Path, min_score: int) -> list[dict]:
    try:
        if fp.stat().st_size > MAX_BYTES:
            return []
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    hits, last = [], -10_000
    for i, line in enumerate(lines):
        if not CALL.search(line) or i - last < 5:           # one hit per call site
            continue
        last = i
        window = "\n".join(call_window(lines, i))
        score, found, votes = 0, [], {"noul": 0, "choice": 0, "score": 0}
        for name, rx, w, qtype in SIGNALS:
            if rx.search(window):
                score += w
                found.append(name)
                if qtype:
                    votes[qtype] += w
        if score < min_score:
            continue
        guess = max(votes, key=votes.get) if any(votes.values()) else "unknown"
        hits.append({"file": str(fp), "line": i + 1, "score": score, "signals": found, "guess": guess,
                     "call": line.strip()[:160]})
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["."])
    ap.add_argument("--min-score", type=int, default=3, help="minimum signal score to report (default 3)")
    ap.add_argument("--json", metavar="PATH", help="also write the hits as JSON")
    args = ap.parse_args(argv)
    hits = []
    for fp in iter_files(args.paths):
        hits.extend(scan_file(fp, args.min_score))
    hits.sort(key=lambda h: (-h["score"], h["file"], h["line"]))
    if not hits:
        print("no decision-shaped LLM calls found (try --min-score 1, or point it at the source folder)")
    for h in hits:
        print(f"{h['file']}:{h['line']}  score={h['score']}  guess={h['guess']}  [{', '.join(h['signals'])}]")
        print(f"    {h['call']}")
    if args.json:
        Path(args.json).write_text(json.dumps(hits, indent=1), encoding="utf-8")
    print(f"\n{len(hits)} candidate call site(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
