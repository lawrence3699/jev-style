"""Demo discovery (``web/demos/<slug>/demo.json``) and extension loading (``ext/<slug>.py``).

demo.json (all keys required)::

    {"slug": "<same as the folder name>", "title_en": str, "title_zh": str,
     "summary_en": str, "summary_zh": str, "order": number, "entry": "index.html"}

``GET /api/demos`` returns ``{"demos": [demo + {"url": "/demos/<slug>/<entry>"}], "errors": [...]}``
sorted by (order, slug). Invalid demo.json files are skipped and reported in ``errors``.

Extensions: every ``ext/<slug>.py`` (not starting with ``_``) is imported as
``jev_style.ext.<slug>`` and its ``register(app)`` is called once at start-up, before the
static mounts, so its routes take precedence over static files. An extension that fails to import
or register is skipped and reported in ``app.state.ext_errors`` (also in ``GET /healthz``).
Inside ``register`` an extension can use ``app.state.adapter.systemone(body_dict)`` (same dict as
the HTTP response; raises ``jev_style.schema.ApiError``) and ``app.state.require_auth`` (a
FastAPI dependency that enforces the server's bearer key when one is configured).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Any

log = logging.getLogger("jev_style.serve")

SERVE_DIR = Path(__file__).resolve().parent
WEB_DIR = SERVE_DIR / "web"
DEMOS_DIR = WEB_DIR / "demos"
EXT_DIR = SERVE_DIR / "ext"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
DEMO_KEYS = {"slug": str, "title_en": str, "title_zh": str, "summary_en": str, "summary_zh": str,
             "order": (int, float), "entry": str}


def _check_demo(folder: Path, meta: Any) -> str | None:
    if not isinstance(meta, dict):
        return "demo.json must be an object"
    for key, typ in DEMO_KEYS.items():
        if key not in meta:
            return f"missing {key}"
        if not isinstance(meta[key], typ) or isinstance(meta[key], bool):
            return f"{key} has the wrong type"
    if meta["slug"] != folder.name:
        return f"slug {meta['slug']!r} does not match folder {folder.name!r}"
    if not SLUG_RE.match(meta["slug"]):
        return "slug must match [a-z0-9][a-z0-9_-]*"
    entry = meta["entry"]
    if entry.startswith("/") or ".." in Path(entry).parts:
        return "entry must be a relative path inside the demo folder"
    if not (folder / entry).is_file():
        return f"entry {entry!r} not found"
    return None


def discover_demos(demos_dir: Path = DEMOS_DIR) -> dict:
    demos, errors = [], []
    if demos_dir.is_dir():
        for folder in sorted(p for p in demos_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
            f = folder / "demo.json"
            if not f.exists():
                continue
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
            except (ValueError, OSError) as e:
                errors.append({"slug": folder.name, "error": f"demo.json unreadable: {e}"})
                continue
            problem = _check_demo(folder, meta)
            if problem:
                errors.append({"slug": folder.name, "error": problem})
                continue
            demos.append({**meta, "url": f"/demos/{meta['slug']}/{meta['entry']}"})
    demos.sort(key=lambda d: (d["order"], d["slug"]))
    return {"demos": demos, "errors": errors}


def load_extensions(app: Any, ext_dir: Path = EXT_DIR, package: str = "jev_style.ext") -> tuple[list[str], list]:
    loaded, errors = [], []
    if not ext_dir.is_dir():
        return loaded, errors
    for f in sorted(ext_dir.glob("*.py")):
        slug = f.stem
        if slug.startswith("_"):
            continue
        name = f"{package}.{slug}"
        try:
            mod = sys.modules.get(name)
            if mod is None or getattr(mod, "__file__", None) != str(f):
                spec = importlib.util.spec_from_file_location(name, f)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            register = getattr(mod, "register", None)
            if not callable(register):
                raise TypeError("module has no register(app)")
            register(app)
            loaded.append(slug)
        except Exception as e:           # one broken demo must not take the server down
            sys.modules.pop(name, None)
            log.error("extension %s failed: %s", slug, e)
            errors.append({"slug": slug, "error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc(limit=3)})
    return loaded, errors
