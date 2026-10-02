"""``jev-style`` command line.

    jev-style serve                      start the local server (API + Playground) on :8765
    jev-style serve --cascade c.json     serve a confidence cascade of several models (see jev_style.cascade)
    jev-style decide "text" --noul "Is this about billing?"
    jev-style download [--backend mlx]   fetch the weights ahead of time
    jev-style guard [--check CMD]        Claude Code PreToolUse hook
    jev-style guard-replay               score the guard on labelled tool calls
    jev-style mcp                        MCP server (stdio) for Claude Code, Cursor, Codex ...
    jev-style eval data.jsonl            accuracy, calibration and automation rate on your labels
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from . import __version__
from .models import BACKENDS, DEFAULT_RELEASE, release_keys

DEFAULT_PORT = 8765


def _release_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--release", default=os.environ.get("JEV_STYLE_RELEASE", DEFAULT_RELEASE), choices=release_keys(),
                    help=f"which model to load (default $JEV_STYLE_RELEASE or {DEFAULT_RELEASE})")


def _model_args(ap: argparse.ArgumentParser) -> None:
    _release_arg(ap)
    ap.add_argument("--backend", default=os.environ.get("JEV_STYLE_BACKEND", "auto"), choices=BACKENDS,
                    help="auto = mlx on Apple silicon when installed, else torch ($JEV_STYLE_BACKEND)")
    ap.add_argument("--model-dir", default=os.environ.get("JEV_STYLE_MODEL_DIR"),
                    help="use a local copy of the model repository instead of downloading ($JEV_STYLE_MODEL_DIR)")
    ap.add_argument("--device", choices=("cuda", "mps", "cpu"),
                    help="torch device (default: $JEV_STYLE_DEVICE, else the best available)")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"), help="torch dtype")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "8bit"), help="mlx weights")
    ap.add_argument("--quant", default="Q8_0", choices=("F16", "Q8_0", "Q4_K_M"), help="gguf file")
    ap.add_argument("--scorer", help="gguf backend: path to the release's scorer binary (0.8b: jev-score, default "
                                      "$JEV_SCORE_BIN; 2b: jev-score-v2, default $JEV_SCORE_V2_BIN)")
    ap.add_argument("--fake", action="store_true", help="deterministic fake engine, no model (tests / UI work)")
    ap.add_argument("--cascade", metavar="CASCADE_JSON",
                    help="a confidence cascade file: its tiers name their own models (--release / --model-dir do not "
                         "apply; --backend, --device, --dtype, --precision, --quant are defaults for local tiers)")


def _load_kw(args: argparse.Namespace) -> dict[str, Any]:
    return {"release": args.release, "model_dir": args.model_dir, "device": args.device, "dtype": args.dtype,
            "precision": args.precision, "quant": args.quant, "scorer": args.scorer}


def _tier_kw(args: argparse.Namespace) -> dict[str, Any]:
    """Loader defaults for a cascade's local tiers (the release, folder and scorer are per tier)."""
    return {"device": args.device, "dtype": args.dtype, "precision": args.precision, "quant": args.quant}


def _cascade_conflict(args: argparse.Namespace) -> str | None:
    if args.cascade and (args.fake or args.model_dir):
        return "--cascade takes its models from the cascade file; drop --fake / --model-dir"
    return None


def _print_tiers(a: Any) -> None:
    for t in a.tier_info():
        keep = f"keep if confidence >= {t['threshold']}" if t["threshold"] is not None else "top tier"
        print(f"  tier {t['tier']}: {t['model']} via {t['target']} ({t['backend']}), {keep}", flush=True)


def cmd_serve(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            print(f"error: environment variable {args.api_key_env} is empty or unset", file=sys.stderr)
            return 2
    conflict = _cascade_conflict(args)
    if conflict:
        print(f"error: {conflict}", file=sys.stderr)
        return 2
    from .server import build_app
    if args.cascade:
        print(f"loading the cascade {args.cascade} (the first run downloads its local models) ...", flush=True)
        app = build_app(args.backend, api_key=api_key, cascade=args.cascade, **_tier_kw(args))
    else:
        print("loading the model (the first run downloads it) ..." if not args.fake else "fake engine", flush=True)
        app = build_app(args.backend, fake=args.fake, api_key=api_key, **({} if args.fake else _load_kw(args)))
    a = app.state.adapter
    print(f"Jev-Style server: model={a.model_id} backend={a.backend} context={a.max_len} "
          f"auth={'on' if api_key else 'off'} -> http://{args.host}:{args.port}/", flush=True)
    if args.cascade:
        _print_tiers(a)
    for e in app.state.ext_errors:
        print(f"warning: extension {e['slug']} not loaded: {e['error']}", file=sys.stderr)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .models import download
    print(download(args.backend, precision=args.precision, quant=args.quant, release=args.release))
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    from .client import JevStyle, JevStyleError, choice, noul, score
    state: Any = sys.stdin.read() if args.state == "-" else args.state
    if args.json_state:
        state = json.loads(state)
    qs: dict[str, dict] = {}
    for i, text in enumerate(args.noul or []):
        qs[f"noul_{i + 1}"] = noul(text)
    for i, spec in enumerate(args.choice or []):
        text, _, opts = spec.partition("::")
        qs[f"choice_{i + 1}"] = choice(text, [o.strip() for o in opts.split(",") if o.strip()])
    for i, spec in enumerate(args.score or []):
        text, _, levels = spec.partition("::")
        qs[f"score_{i + 1}"] = score(text, [x.strip() for x in levels.split(",") if x.strip()])
    if not qs:
        print("error: give at least one --noul, --choice or --score", file=sys.stderr)
        return 2
    conflict = _cascade_conflict(args)
    if conflict:
        print(f"error: {conflict}", file=sys.stderr)
        return 2
    url = args.url or os.environ.get("JEV_STYLE_URL")
    if args.cascade and not args.url:
        js = JevStyle(cascade=args.cascade, backend=args.backend, **_tier_kw(args))
    else:
        js = JevStyle(base_url=url) if url else JevStyle(backend=args.backend, fake=args.fake,
                                                         **({} if args.fake else _load_kw(args)))
    try:
        out = js.decide(state, qs)
    except JevStyleError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_guard(argv: list[str]) -> int:
    from . import guard
    try:
        return guard.main(argv)
    except SystemExit:
        raise
    except Exception as exc:  # last resort: ask, never block or allow by accident
        sys.stdout.write(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "ask",
            "permissionDecisionReason": f"Jev-Style guard failed ({exc.__class__.__name__}); asking instead."}}) + "\n")
        return 0


def main(argv: list[str] | None = None) -> int:
    from .cascade import CascadeConfigError
    from .models import MissingBackendError
    try:
        return _main(argv)
    except (MissingBackendError, CascadeConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # subcommands with their own parsers
    if argv and argv[0] == "guard":
        return cmd_guard(argv[1:])
    if argv and argv[0] == "guard-replay":
        from .guard_replay import main as replay
        return replay(argv[1:])
    if argv and argv[0] == "eval":
        from .evaluate import main as ev
        return ev(argv[1:])
    if argv and argv[0] == "mcp":
        from .mcp_server import main as mcp
        return mcp(argv[1:])

    ap = argparse.ArgumentParser(prog="jev-style", description="Small, calibrated, local decision models.",
                                 epilog="also: jev-style eval | guard | guard-replay | mcp  (each has its own --help)")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the local server (API + Playground)")
    _model_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=int(os.environ.get("JEV_STYLE_PORT", DEFAULT_PORT)),
                   help=f"default $JEV_STYLE_PORT or {DEFAULT_PORT}")
    s.add_argument("--api-key-env", metavar="NAME", help="require 'Authorization: Bearer <key>', key from env NAME")
    s.set_defaults(func=cmd_serve)

    d = sub.add_parser("download", help="download the weights for a backend")
    _release_arg(d)
    d.add_argument("--backend", default=os.environ.get("JEV_STYLE_BACKEND", "auto"), choices=BACKENDS)
    d.add_argument("--precision", default="bf16", choices=("bf16", "8bit"))
    d.add_argument("--quant", default="Q8_0", choices=("F16", "Q8_0", "Q4_K_M"))
    d.set_defaults(func=cmd_download)

    q = sub.add_parser("decide", help="answer questions about a text from the command line")
    q.add_argument("state", help="the text to judge ('-' reads stdin)")
    q.add_argument("--json-state", action="store_true", help="parse the state as JSON")
    q.add_argument("--noul", action="append", metavar="STATEMENT", help="yes/no question (repeatable)")
    q.add_argument("--choice", action="append", metavar="Q::a,b,c", help="multiple choice (repeatable)")
    q.add_argument("--score", action="append", metavar="Q::low,mid,high", help="ordered scale, lowest first")
    q.add_argument("--url", help="use a running server instead of loading the model ($JEV_STYLE_URL)")
    _model_args(q)
    q.set_defaults(func=cmd_decide)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
