#!/usr/bin/env python3
"""Jev-Style MCP server: typed decisions (yes/no, choice, score) as MCP tools.

Tools
  decide(state, questions)          full systemone-compatible request, several questions at once
  noul(state, question, ...)        probability that a statement about the state is true
  choice(state, question, options)  pick one option, with probabilities and confidence
  score(state, question, levels)    place the state on an ordered scale of 2-10 levels
  model_info()                      which model and limits the backend serves

Backends (pick one)
  --url URL        a running ``jev-style serve`` (default http://127.0.0.1:8765, or $JEV_STYLE_URL)
  --model          the real model in-process (downloads it once; --backend auto|torch|mlx|gguf)
  --fake           the fake engine in-process (no model; for tests and demos)

    jev-style mcp                                   # talks to a running server
    claude mcp add jev-style -- jev-style mcp       # register with Claude Code

Transport is stdio unless ``--transport streamable-http`` is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError as _e:  # pragma: no cover
    raise SystemExit("the MCP server needs the 'mcp' extra: pip install 'jev-style[mcp]'") from _e

from . import __version__  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8765"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}

INSTRUCTIONS = """Jev-Style is a small local decision model. It does not write text: it reads a state
(text or JSON, up to 25,600 tokens) and answers typed questions about it with calibrated probabilities.
- noul: probability that a statement is true (0..1). Use for checks and gates.
- choice: one option out of 1..255, with a probability per option and a confidence.
- score: position on an ordered scale of 2..10 levels (lowest first); score = expected level index.
Ask several questions about the same state in one `decide` call: the state is read once.
Each question with its options must fit in 2,048 tokens; nothing is truncated, oversize input is an error.
Turn probabilities into actions with thresholds in code (e.g. act >= 0.9, review 0.5-0.9, stop < 0.5).
Everything runs on this machine; nothing is sent elsewhere."""

StateT = Union[str, Dict[str, Any], List[Any]]


class BackendError(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# backends: each is a callable body -> response dict (the HTTP response body of /v1/systemone)
# ---------------------------------------------------------------------------------------------
class HttpBackend:
    def __init__(self, url: str, api_key: Optional[str] = None, timeout: float = 120.0, allow_remote: bool = False):
        self.url = url.rstrip("/")
        host = (urllib.parse.urlparse(self.url).hostname or "").lower()
        if host not in LOOPBACK and not host.startswith("127.") and not allow_remote:
            raise BackendError(f"refusing non-local Jev-Style URL {url!r} (pass --allow-remote to override)")
        self.api_key = api_key
        self.timeout = timeout
        self.description = f"http {self.url}"

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (local URL)
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read().decode("utf-8")).get("error", {})
            except Exception:
                err = {}
            q = f" (question {err['question']!r})" if err.get("question") else ""
            raise BackendError(f"{e.code} {err.get('code', 'http_error')}: {err.get('message', e.reason)}{q}")
        except (urllib.error.URLError, OSError) as e:
            raise BackendError(f"Jev-Style server not reachable at {self.url}: {getattr(e, 'reason', e)}. "
                               f"Start it with: jev-style serve")

    def decide(self, body: dict) -> dict:
        return self._req("POST", "/v1/systemone", body)

    def models(self) -> dict:
        return self._req("GET", "/v1/models")


class InProcessBackend:
    def __init__(self, fake: bool = False, backend: str = "auto"):
        from .client import JevStyle
        self._mj = JevStyle(fake=fake, backend=backend)
        self.description = "in-process fake engine" if fake else f"in-process model ({backend})"

    def decide(self, body: dict) -> dict:
        from .client import JevStyleError
        try:
            return self._mj.decide(body["state"], body["questions"], model=body.get("model"))
        except JevStyleError as e:
            q = f" (question {e.question!r})" if e.question else ""
            raise BackendError(f"{e.status} {e.code}: {e.message}{q}")

    def models(self) -> dict:
        return self._mj.models()


# ---------------------------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------------------------
def build_server(backend: Any) -> MCPServer:
    mcp = MCPServer("jev-style", title="Jev-Style decisions", instructions=INSTRUCTIONS, version=__version__)

    def run(state: Any, questions: Dict[str, Any]) -> dict:
        try:
            return backend.decide({"state": state, "questions": questions})
        except BackendError as e:
            raise ToolError(str(e)) from None

    def meta(resp: dict) -> dict:
        return {"model": resp.get("model"), "input_tokens": (resp.get("usage") or {}).get("input_tokens"),
                "total_ms": (resp.get("timing") or {}).get("total_ms")}

    @mcp.tool()
    def decide(state: StateT, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Ask several typed questions about one state in a single call (the state is read once).

        state: text, or a JSON object/array (serialised for the model).
        questions: {id: question}. A question is
          {"type": "noul", "instructions": "<statement or yes/no question>", "criteria"?: {"true": "...", "false": "..."}}
          {"type": "choice", "instructions": "...", "criteria": {"<option>": "<description or null>", ...}}
          {"type": "score", "instructions": "...", "criteria": ["<lowest level>", ..., "<highest level>"]}
          Score levels can also be {"label": "...", "description": "..."}.
        Returns the systemone-compatible response: answers per id (noul -> p_true; choice -> choice,
        probabilities, confidence; score -> score, legend, probabilities, confidence), usage and timing.
        """
        if not isinstance(questions, dict) or not questions:
            raise ToolError("questions must be a non-empty object {id: question}")
        return run(state, questions)

    @mcp.tool()
    def noul(state: StateT, question: str, true_means: Optional[str] = None,
             false_means: Optional[str] = None) -> Dict[str, Any]:
        """Probability that a yes/no statement about the state is true.

        question: e.g. "The customer asks for a refund." or "Is the invoice overdue?"
        true_means / false_means: optional descriptions of what counts as true / false.
        Returns {"p_true": 0..1, "answer": p_true >= 0.5, ...}. Gate with your own thresholds.
        """
        q: Dict[str, Any] = {"type": "noul", "instructions": question}
        crit = {k: v for k, v in (("true", true_means), ("false", false_means)) if v}
        if crit:
            q["criteria"] = crit
        resp = run(state, {"q": q})
        p = float(resp["answers"]["q"]["noul"])
        return {"p_true": p, "answer": p >= 0.5, **meta(resp)}

    @mcp.tool()
    def choice(state: StateT, question: str, options: Union[List[str], Dict[str, Optional[str]]]) -> Dict[str, Any]:
        """Pick the best option for the state.

        options: a list of option names, or {option: description or null} (1..255 options, order kept).
        Returns {"choice", "probabilities": {option: p}, "confidence": 0..1, ...}.
        confidence = (k * p_max - 1) / (k - 1) for k options: 0 = uniform, 1 = certain.
        """
        crit = {str(o): None for o in options} if isinstance(options, list) else dict(options)
        if not crit:
            raise ToolError("options must not be empty")
        resp = run(state, {"q": {"type": "choice", "instructions": question, "criteria": crit}})
        a = resp["answers"]["q"]
        return {"choice": a["choice"], "probabilities": a["probabilities"], "confidence": a["confidence"], **meta(resp)}

    @mcp.tool()
    def score(state: StateT, question: str, levels: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Place the state on an ordered scale.

        levels: 2..10 levels, lowest first; each a label or {"label": ..., "description": ...}.
        Returns {"score": expected level index (0 = first level, may fall between levels),
        "level": label of the most likely level, "legend": {index: label}, "probabilities": {index: p},
        "confidence", ...}.
        """
        if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
            raise ToolError("levels must be a list of 2 to 10 levels, lowest first")
        resp = run(state, {"q": {"type": "score", "instructions": question, "criteria": levels}})
        a = resp["answers"]["q"]
        probs = a["probabilities"]
        top = max(probs, key=lambda k: probs[k])
        return {"score": a["score"], "level": a["legend"].get(top, top), "legend": a["legend"],
                "probabilities": probs, "confidence": a["confidence"], **meta(resp)}

    @mcp.tool()
    def model_info() -> Dict[str, Any]:
        """Model id, context and question budgets, and which backend this MCP server uses."""
        try:
            info = backend.models()
        except BackendError as e:
            raise ToolError(str(e)) from None
        return {"backend": backend.description, **info}

    return mcp


def make_backend(args: argparse.Namespace) -> Any:
    if args.fake:
        return InProcessBackend(fake=True)
    if args.model:
        return InProcessBackend(backend=args.backend)
    url = args.url or os.environ.get("JEV_STYLE_URL") or DEFAULT_URL
    return HttpBackend(url, api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
                       timeout=args.timeout, allow_remote=args.allow_remote)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="jev-style mcp", description="Jev-Style MCP server")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--url", help=f"Jev-Style server URL (default $JEV_STYLE_URL or {DEFAULT_URL})")
    src.add_argument("--model", action="store_true", help="load the real model in-process")
    src.add_argument("--fake", action="store_true", help="in-process fake engine, no model")
    ap.add_argument("--backend", default="auto", choices=("auto", "torch", "mlx", "gguf"))
    ap.add_argument("--api-key-env", default="JEV_STYLE_API_KEY", help="env var holding the server's bearer key")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--allow-remote", action="store_true", help="allow a non-loopback --url")
    ap.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    ap.add_argument("--port", type=int, default=8766, help="port for streamable-http")
    args = ap.parse_args(argv)
    try:
        backend = make_backend(args)
    except BackendError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    server = build_server(backend)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
