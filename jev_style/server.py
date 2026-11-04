"""FastAPI app: systemone-compatible API + Playground + demos.

Routes::

    POST /v1/systemone      decisions (bearer auth when an API key is configured)
    GET  /v1/models         model list (bearer auth when an API key is configured)
    GET  /healthz           liveness + backend + extension status (no auth)
    GET  /api/demos         discovered demos (no auth)
    /demos/<slug>/...       static files from web/demos/<slug>/
    /demo-data/<slug>/...   read-only demo data from web/demo-data/<slug>/
    /                       web/ (index.html = Playground, common.js, common.css)

Cross-site requests are refused (403 ``forbidden_host`` / ``forbidden_origin``): by default the server answers
only requests whose ``Host`` is a loopback name (plus the name it binds to, see ``jev-style serve --host``) and
whose ``Origin``, when a browser sends one, is loopback or the same host. A web page on another site cannot use a
local server as its API, with or without DNS rebinding; the Playground and apps on localhost are unaffected.
``create_app(allowed_hosts=None)`` / ``jev-style serve --allow-any-host`` switch the check off.
"""
from __future__ import annotations

import hmac
import logging
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, discovery
from .adapter import Adapter
from .assets import header_on, split_multipart
from .schema import MAX_BODY_BYTES, ApiError, decode_body, parse_json

log = logging.getLogger("jev_style.serve")
DEMO_DATA_DIR = discovery.WEB_DIR / "demo-data"
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def header_host(value: str) -> str:
    """The host name of a ``Host`` header (``name[:port]``, ``[v6]:port``) or of an ``Origin`` URL, lower-cased."""
    v = (value or "").strip().lower()
    if "://" in v:
        try:
            return urlparse(v).hostname or ""
        except ValueError:
            return ""
    if v.startswith("["):
        return v[1:v.index("]")] if "]" in v else v[1:]
    return v.split(":", 1)[0]


def _is_loopback(name: str) -> bool:
    return name in LOOPBACK_NAMES or name.startswith("127.")


def request_problem(host_header: str, origin_header: str | None, allowed: Any) -> tuple[str, str] | None:
    """(code, message) when a request must be refused, else None. ``allowed``: None = accept anything;
    ``"bound"`` = any Host (the server binds to every interface), but a browser ``Origin`` must be loopback or
    the request's own host; a collection of names = the Host must be one of them (or loopback) and so must
    the Origin."""
    if allowed is None:
        return None
    host = header_host(host_header)
    names = None if allowed == "bound" else {str(n).lower() for n in allowed}
    if names is not None and not (host in names or _is_loopback(host)):
        return ("forbidden_host", f"this server answers requests addressed to {', '.join(sorted(names))} only, "
                                  f"not to {host!r} (jev-style serve --host NAME, or --allow-any-host)")
    if origin_header is None:
        return None
    origin = header_host(origin_header)
    if origin_header.strip().lower() == "null" or not origin:
        return ("forbidden_origin", "requests from an opaque (null) origin are refused")
    if origin == host or _is_loopback(origin) or (names is not None and origin in names):
        return None
    return ("forbidden_origin", f"cross-site requests from {origin!r} are refused: this is a local server")


async def _read_body(request: Request) -> bytes:
    """The raw body, refused with 413 as soon as it passes ``MAX_BODY_BYTES`` (before decompression)."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise ApiError(413, "payload_too_large", f"request body is over {MAX_BODY_BYTES // 2**20} MiB")
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > MAX_BODY_BYTES:
            raise ApiError(413, "payload_too_large", f"request body is over {MAX_BODY_BYTES // 2**20} MiB")
    return bytes(buf)


class _Static(StaticFiles):
    """StaticFiles that answers 404 (not 500) while its directory does not exist yet."""

    async def __call__(self, scope, receive, send):
        if self.directory is not None and not Path(self.directory).is_dir():
            raise StarletteHTTPException(status_code=404)
        await super().__call__(scope, receive, send)


def create_app(adapter: Adapter | Any, *, api_key: str | None = None, web_dir: Path = discovery.WEB_DIR,
               demos_dir: Path | None = None, demo_data_dir: Path = DEMO_DATA_DIR,
               ext_dir: Path | None = discovery.EXT_DIR, ext_package: str = "jev_style.ext",
               allowed_hosts: Iterable[str] | str | None = "loopback") -> FastAPI:
    """``allowed_hosts``: the ``Host`` names this server answers to (see ``request_problem``): ``"loopback"``
    (default: 127.0.0.1, localhost, ::1), ``"bound"`` (any Host, same-host or loopback Origin only), a collection
    of names, or None to accept every request."""
    web_dir = Path(web_dir)
    demos_dir = Path(demos_dir) if demos_dir is not None else web_dir / "demos"
    app = FastAPI(title="Jev-Style local server", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.adapter = adapter
    app.state.api_key = api_key
    app.state.allowed_hosts = set(LOOPBACK_NAMES) if allowed_hosts == "loopback" else allowed_hosts

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse({"error": {"code": code, "message": str(exc.detail)}}, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _internal(_: Request, exc: Exception):
        log.exception("internal error")
        return JSONResponse({"error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}},
                            status_code=500)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/v1/"):
            response.headers["x-request-id"] = "req_" + uuid.uuid4().hex
        return response

    @app.middleware("http")
    async def _same_host(request: Request, call_next):
        problem = request_problem(request.headers.get("host", ""), request.headers.get("origin"),
                                  app.state.allowed_hosts)
        if problem:
            return JSONResponse({"error": {"code": problem[0], "message": problem[1]}}, status_code=403)
        return await call_next(request)

    def require_auth(request: Request) -> None:
        key = app.state.api_key
        if not key:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip().encode(), key.encode()):
            raise ApiError(401, "unauthorized", "missing or invalid API key (Authorization: Bearer <key>)")

    app.state.require_auth = require_auth

    @app.post("/v1/systemone", dependencies=[Depends(require_auth)])
    async def systemone(request: Request):
        adapter = app.state.adapter
        images_ok = bool(getattr(adapter, "supports_images", False))
        ctype = request.headers.get("content-type", "")
        if ctype.lower().startswith("multipart/"):
            if not images_ok:
                raise ApiError(400, "images_not_supported", "this model reads text only; multipart requests (images) "
                               "are not supported")
            raw = await _read_body(request)
            body, parts = await run_in_threadpool(split_multipart, raw, ctype)
        else:
            raw = await _read_body(request)
            body, parts = parse_json(await run_in_threadpool(decode_body, raw, request.headers.get("content-encoding"))), None
        if not images_ok:
            return JSONResponse(await run_in_threadpool(adapter.systemone, body))
        return JSONResponse(await run_in_threadpool(lambda: adapter.systemone(body, parts=parts,
                                                                              image_header=header_on(request.headers))))

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def models():
        return app.state.adapter.models()

    @app.get("/healthz")
    async def healthz():
        a = app.state.adapter
        rel = a.release_info()
        return {"status": "ok", "model": a.model_id, "backend": a.backend, "auth": bool(app.state.api_key),
                "untrained": rel["untrained"], "release_kind": rel["release_kind"],
                "extensions": app.state.extensions, "ext_errors": [
                    {"slug": e["slug"], "error": e["error"]} for e in app.state.ext_errors]}

    @app.get("/api/demos")
    async def demos():
        return discovery.discover_demos(demos_dir)

    loaded, errors = discovery.load_extensions(app, ext_dir, ext_package) if ext_dir else ([], [])
    app.state.extensions, app.state.ext_errors = loaded, errors

    # static mounts last: API and extension routes win
    app.mount("/demos", _Static(directory=demos_dir, html=True, check_dir=False), name="demos")
    app.mount("/demo-data", _Static(directory=demo_data_dir, check_dir=False), name="demo-data")
    app.mount("/", _Static(directory=web_dir, html=True, check_dir=False), name="web")
    return app


def build_app(backend: str = "auto", *, fake: bool = False, api_key: str | None = None, cascade: Any = None,
              allowed_hosts: Iterable[str] | str | None = "loopback", **load_kw: Any) -> FastAPI:
    """Load the model (or the fake engine, or every tier of a cascade file) and return the app."""
    if cascade is not None:
        if fake:
            raise ValueError("cascade and fake cannot be combined (use 'fake' tier targets instead)")
        from .cascade import CascadeAdapter
        return create_app(CascadeAdapter.from_config(cascade, backend=backend, **load_kw), api_key=api_key,
                          allowed_hosts=allowed_hosts)
    from .adapter import build_adapter
    return create_app(build_adapter(backend, fake=fake, **load_kw), api_key=api_key, allowed_hosts=allowed_hosts)
