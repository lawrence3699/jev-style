"""FastAPI app: systemone-compatible API + Playground + demos.

Routes::

    POST /v1/systemone      decisions (bearer auth when an API key is configured)
    GET  /v1/models         model list (bearer auth when an API key is configured)
    GET  /healthz           liveness + backend + extension status (no auth)
    GET  /api/demos         discovered demos (no auth)
    /demos/<slug>/...       static files from web/demos/<slug>/
    /demo-data/<slug>/...   read-only demo data from web/demo-data/<slug>/
    /                       web/ (index.html = Playground, common.js, common.css)
"""
from __future__ import annotations

import hmac
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, discovery
from .adapter import Adapter
from .schema import ApiError, parse_json

log = logging.getLogger("jev_style.serve")
DEMO_DATA_DIR = discovery.WEB_DIR / "demo-data"


class _Static(StaticFiles):
    """StaticFiles that answers 404 (not 500) while its directory does not exist yet."""

    async def __call__(self, scope, receive, send):
        if self.directory is not None and not Path(self.directory).is_dir():
            raise StarletteHTTPException(status_code=404)
        await super().__call__(scope, receive, send)


def create_app(adapter: Adapter, *, api_key: str | None = None, web_dir: Path = discovery.WEB_DIR,
               demos_dir: Path | None = None, demo_data_dir: Path = DEMO_DATA_DIR,
               ext_dir: Path | None = discovery.EXT_DIR, ext_package: str = "jev_style.ext") -> FastAPI:
    web_dir = Path(web_dir)
    demos_dir = Path(demos_dir) if demos_dir is not None else web_dir / "demos"
    app = FastAPI(title="Jev-Style local server", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.adapter = adapter
    app.state.api_key = api_key

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
        body = parse_json(await request.body())
        return JSONResponse(await run_in_threadpool(app.state.adapter.systemone, body))

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


def build_app(backend: str = "auto", *, fake: bool = False, api_key: str | None = None, **load_kw: Any) -> FastAPI:
    """Load the model (or the fake engine) and return the app."""
    from .adapter import build_adapter
    return create_app(build_adapter(backend, fake=fake, **load_kw), api_key=api_key)
