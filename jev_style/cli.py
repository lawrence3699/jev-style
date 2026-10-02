"""``jev-style`` command line.

    jev-style serve                      start the local server (API + Playground) on :8765
    jev-style serve --cascade c.json     serve a confidence cascade of several models (see jev_style.cascade)
    jev-style serve --cascade cascade-9b serve a named cascade (jev-style releases lists them)
    jev-style decide "text" --noul "Is this about billing?"
    jev-style download [--backend mlx]   fetch the weights ahead of time (--cascade NAME|FILE: every tier)
    jev-style releases [--json]          the model releases and named cascades this version pins
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
    ap.add_argument("--cuda-graphs", default="auto", choices=("auto", "on", "off"),
                    help="torch on CUDA: record CUDA graphs at start-up for much faster short inputs (auto = on when "
                         "the model runs on CUDA, or $JEV_STYLE_CUDA_GRAPHS)")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "8bit"), help="mlx weights")
    ap.add_argument("--quant", default="Q8_0", choices=("F16", "Q8_0", "Q4_K_M"), help="gguf file")
    ap.add_argument("--scorer", help="gguf backend: path to the release's scorer binary (0.8b: jev-score, default "
                                      "$JEV_SCORE_BIN; 2b: jev-score-v2, default $JEV_SCORE_V2_BIN)")
    ap.add_argument("--fake", action="store_true", help="deterministic fake engine, no model (tests / UI work)")
    ap.add_argument("--cascade", metavar="NAME_OR_FILE",
                    help="a confidence cascade: a name from `jev-style releases` (e.g. cascade-9b) or a cascade file. "
                         "Its tiers name their own models (--release / --model-dir do not apply; --backend, --device, "
                         "--dtype, --cuda-graphs, --precision, --quant are defaults for local tiers)")


def _cuda_graphs(args: argparse.Namespace) -> bool | None:
    return {"auto": None, "on": True, "off": False}[args.cuda_graphs]


def _load_kw(args: argparse.Namespace) -> dict[str, Any]:
    return {"release": args.release, "model_dir": args.model_dir, "device": args.device, "dtype": args.dtype,
            "precision": args.precision, "quant": args.quant, "scorer": args.scorer, "cuda_graphs": _cuda_graphs(args)}


def _tier_kw(args: argparse.Namespace) -> dict[str, Any]:
    """Loader defaults for a cascade's local tiers (the release, folder and scorer are per tier)."""
    return {"device": args.device, "dtype": args.dtype, "precision": args.precision, "quant": args.quant,
            "cuda_graphs": _cuda_graphs(args)}


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
        print(f"loading the cascade {args.cascade} (the first run downloads its models) ...", flush=True)
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
    if args.cascade:
        from .cascade import download_tiers
        for tier, folder in download_tiers(args.cascade, backend=args.backend, precision=args.precision,
                                           quant=args.quant):
            where = folder if folder is not None else f"nothing to download ({tier.target})"
            print(f"{tier.name}\t{where}", flush=True)
        return 0
    from .models import download
    print(download(args.backend, precision=args.precision, quant=args.quant, release=args.release))
    return 0


def cmd_releases(args: argparse.Namespace) -> int:
    from . import cascades
    from .models import ALIASES, DEFAULT_RELEASE, RELEASES
    if args.json:
        print(json.dumps({"version": __version__, "default_release": DEFAULT_RELEASE, "releases": {
            k: {"model_id": r.model_id, "title": r.title, "release_date": r.release_date, "published": r.published,
                "aliases": [a for a, key in ALIASES.items() if key == k],
                "builds": {b.backend: {"repo": b.repo, "revision": b.revision} for b in r.builds.values()}}
            for k, r in RELEASES.items()},
            "cascades": {n: {**cascades.CASCADES[n], "placeholders": cascades.placeholders(n)}
                         for n in cascades.names()}}, indent=2))
        return 0
    print(f"jev-style {__version__}\n\nmodel releases (--release):")
    for k, r in RELEASES.items():
        tags = [a for a, key in ALIASES.items() if key == k] + (["default"] if k == DEFAULT_RELEASE else [])
        state = r.release_date if r.published else "not published in this version"
        print(f"  {k:<9} {r.model_id}  ({', '.join(tags)}; {state})")
        for b in r.builds.values():
            print(f"      {b.backend:<6} {b.repo} @ {b.revision[:8] if b.revision else 'unpinned'}")
    print("\nnamed cascades (--cascade NAME):")
    for n in cascades.names():
        d, missing = cascades.CASCADES[n], cascades.placeholders(n)
        state = f"NOT FROZEN in this version, placeholder: {', '.join(missing)}" if missing else \
            f"frozen {d['frozen_utc']}"
        print(f"  {n:<11} {d['id']}  ({state})")
        th = d["thresholds"]
        for i, t in enumerate(d["tiers"]):
            rule = "top tier" if i >= len(th) else f"keep if confidence >= {'?' if th[i] is None else th[i]}"
            rev = f" @ {t['revision'][:8]}" if t.get("revision") else ""
            print(f"      tier {i + 1}: {t['model_id']} via {t['target']}{rev}, {rule}")
        if d.get("calibration_sha256"):
            print(f"      calibration set sha256 {d['calibration_sha256']}")
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
    from .jevk5_engine import JevK5LoadError
    from .models import MissingBackendError
    try:
        return _main(argv)
    except (MissingBackendError, CascadeConfigError, JevK5LoadError) as e:
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
    d.add_argument("--cascade", metavar="NAME_OR_FILE",
                   help="download every tier of a cascade instead (--release does not apply; --backend, --precision, "
                        "--quant are defaults for local tiers)")
    d.set_defaults(func=cmd_download)

    r = sub.add_parser("releases", help="list the model releases and named cascades this version pins")
    r.add_argument("--json", action="store_true", help="machine-readable, with each named cascade's full config")
    r.set_defaults(func=cmd_releases)

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
