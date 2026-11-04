"""Python client for Jev-Style: in-process, or against any systemone-compatible server.

    from jev_style import JevStyle, noul, choice, score
    js = JevStyle(fake=True)                             # in-process fake engine (no model)
    js = JevStyle(backend="auto")                        # in-process real model (downloads it once)
    js = JevStyle(release="2b")                          # another release (see jev_style.models.RELEASES)
    js = JevStyle.from_pretrained("chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF")   # pick the build by repo id
    js = JevStyle(base_url="http://127.0.0.1:8765")      # over HTTP: jev-style serve, or another server
                                                         # that implements POST /v1/systemone
    js = JevStyle(base_url="https://api.jevstyle.com", api_key="...")   # a hosted server: pass its key
    js = JevStyle(cascade="cascade.json")                # a confidence cascade of several models
                                                         # (see jev_style.cascade)
    js = JevStyle(cascade="cascade-9b")                  # a named cascade (jev_style.cascades.CASCADES)
    out = js.decide("I was charged twice.", {
        "billing": noul("Is this about billing?"),
        "team": choice("Which team?", {"billing": None, "tech": "bugs, outages"}),
        "phishing": noul("This message is a phishing attempt."),
    })
    out["answers"]["team"]["choice"]

    js = JevStyle(release="2b")                          # images (2B v4): the state refers to them by key
    js.decide({"receipt": "r1"}, {"over": noul("Is the total over 50?")}, assets={"r1": image("receipt.jpg")})

``decide`` returns exactly the HTTP response body of ``POST /v1/systemone`` and raises
``JevStyleError`` (status, code, message, question) for the contract's error bodies. It reads both error
shapes in use: the package's nested ``{"error": {"code", "message", "question"?}}`` and the flat
``{"error": "<code>", "message": "..."}`` of the hosted api.jevstyle.com (``parse_error_body``). Over-long input
is ``400 max_tokens_exceeded`` from this package's server and ``413 input_too_large`` from the hosted API;
``JevStyleError.input_too_long`` is true for both.

The API key: ``api_key=`` if given, else ``$JEV_STYLE_API_KEY``, which is only sent to a loopback server or to
the server named in ``$JEV_STYLE_URL``; any other ``base_url`` needs an explicit ``api_key`` (so a key for the
local server or for api.jevstyle.com is never sent to a third-party server by accident).

For scripts there are module-level shortcuts that share one lazily created client
(``$JEV_STYLE_URL`` if set, else the local model with ``$JEV_STYLE_BACKEND`` or ``auto``)::

    import jev_style
    jev_style.classify("Where is my parcel?", ["billing", "shipping", "tech"])["label"]   # 'shipping'
    jev_style.decide("I was charged twice.", {"billing": noul("This is about billing.")})
    jev_style.configure(base_url="http://127.0.0.1:8765")     # optional: choose the engine yourself
"""
from __future__ import annotations

import os
import threading
from typing import Any, Iterable, Mapping


# Error codes that all mean "the input is over the model's budget": this package's server and TypeSafe's API
# answer 400 ``max_tokens_exceeded``, the hosted api.jevstyle.com 413 ``input_too_large``, servers before jev-style
# 0.5.0 422 ``input_budget_exceeded``.
INPUT_TOO_LONG_CODES = frozenset({"max_tokens_exceeded", "input_too_large", "input_budget_exceeded"})
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class JevStyleError(Exception):
    def __init__(self, status: int, code: str, message: str, question: str | None = None, body: dict | None = None):
        super().__init__(f"{status} {code}: {message}" + (f" (question {question!r})" if question else ""))
        self.status, self.code, self.message, self.question, self.body = status, code, message, question, body or {}

    @property
    def input_too_long(self) -> bool:
        """The input was over the model's budget, whichever server answered (see ``INPUT_TOO_LONG_CODES``)."""
        return self.code in INPUT_TOO_LONG_CODES


def parse_error_body(body: Any, status: int | None = None) -> dict:
    """The ``{"code", "message", "question"?}`` of an error response body, whichever shape the server uses:

    * nested, this package's contract: ``{"error": {"code": ..., "message": ..., "question"?: ...}}``
    * flat, the hosted api.jevstyle.com: ``{"error": "<code>", "message": "...", ...}``
    * a bare message, e.g. jevk5-serve: ``{"error": "<message>"}`` (code ``http_error``)

    Anything else gives ``{"code": "http_error", "message": ""}``; ``status`` only names the code when it is a
    well-known one and the body names none."""
    out: dict = {"code": "http_error", "message": ""}
    if not isinstance(body, dict):
        return _with_status(out, status)
    err = body.get("error")
    if isinstance(err, dict):
        out["code"] = str(err.get("code") or out["code"])
        out["message"] = str(err.get("message") or "")
        if err.get("question") is not None:
            out["question"] = str(err["question"])
        for k in ("limit", "retry_after"):
            if err.get(k) is not None:
                out[k] = err[k]
    elif isinstance(err, str):
        if "message" in body:                   # flat: "error" is the code
            out["code"], out["message"] = err, str(body.get("message") or "")
        else:                                   # a bare message
            out["message"] = err
        if body.get("question") is not None:
            out["question"] = str(body["question"])
    elif isinstance(body.get("message"), str):
        out["message"] = body["message"]
    return _with_status(out, status)


def _with_status(err: dict, status: int | None) -> dict:
    if err["code"] == "http_error" and status == 401:
        err["code"] = "unauthorized"
    return err


def noul(instructions: Any, criteria: Mapping[str, Any] | None = None) -> dict:
    q = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        q["criteria"] = dict(criteria)
    return q


def choice(instructions: Any, options: Mapping[str, Any] | Iterable[str]) -> dict:
    crit = dict(options) if isinstance(options, Mapping) else {str(o): None for o in options}
    return {"type": "choice", "instructions": instructions, "criteria": crit}


def score(instructions: Any, levels: Iterable[Any]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def image(src: Any, format: str | None = None) -> dict:
    """An ``assets`` entry for ``decide(..., assets=...)``: a PNG or JPEG file path or its bytes ->
    {"type": "image", "format": "png" | "jpg", "data_base64": ...}, the form TypeSafe's API takes."""
    import base64
    from pathlib import Path
    data = bytes(src) if isinstance(src, (bytes, bytearray, memoryview)) else Path(src).expanduser().read_bytes()
    if format is None:
        format = "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "jpg" if data.startswith(b"\xff\xd8\xff") else None
        if format is None:
            raise ValueError("not a PNG or JPEG image (the only formats the API takes); pass format= to override")
    return {"type": "image", "format": format, "data_base64": base64.b64encode(data).decode("ascii")}


class JevStyle:
    def __init__(self, base_url: str | None = None, backend: str = "auto", fake: bool = False,
                 api_key: str | None = None, timeout: float = 120.0, http_client: Any = None,
                 cascade: Any = None, adapter: Any = None, **load_kw: Any):
        """``cascade``: a cascade name (``jev-style releases``), file or dict. ``adapter``: an already built engine
        with the Adapter surface (``systemone(body)``, ``models()``), e.g. ``jevk5_engine.load_jevk5(...)``."""
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key if api_key is not None else implicit_api_key(self.base_url)
        self.timeout = timeout
        self._http = http_client
        self._adapter = None
        if adapter is not None or cascade is not None:
            if self.base_url or fake or http_client is not None or (adapter is not None and cascade is not None):
                raise ValueError("cascade / adapter cannot be combined with each other, base_url, http_client or "
                                 "fake")
        if adapter is not None:
            self._adapter = adapter
        elif cascade is not None:           # a cascade name / file / dict; backend and load_kw go to its local tiers
            from .cascade import CascadeAdapter
            self._adapter = CascadeAdapter.from_config(cascade, backend=backend, **load_kw)
        elif self.base_url is None and http_client is None:
            from .adapter import build_adapter
            self._adapter = build_adapter(backend, fake=fake, **load_kw)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, trust_remote_code: bool = False, **load_kw: Any) -> "JevStyle":
        """Load a Jev-Style release in-process by its Hub repo id; the repo decides the backend (a ``-GGUF`` repo
        runs on llama.cpp, ``-MLX`` on MLX, the main repo on PyTorch). Keyword arguments go to the loader
        (``precision="8bit"``, ``quant="Q4_K_M"``, ``device="cpu"``, ``model_dir=...``)."""
        return cls(repo=repo_id, trust_remote_code=trust_remote_code, **load_kw)

    # -- transport ---------------------------------------------------------------------------
    def _client(self):
        if self._http is None:
            import httpx
            self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._http

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @staticmethod
    def _raise(resp) -> None:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        err = parse_error_body(body, resp.status_code)
        raise JevStyleError(resp.status_code, err["code"], err["message"] or resp.text[:500], err.get("question"),
                          body if isinstance(body, dict) else {})

    # -- API ---------------------------------------------------------------------------------
    def decide(self, state: Any, questions: Mapping[str, dict], model: str | None = None,
               assets: Mapping[str, Any] | None = None) -> dict:
        """``assets``: images that string values in ``state`` refer to by key, e.g. {"r1": image("receipt.jpg")}
        (models that read images: 2B v4). Sent the way TypeSafe's API takes them, with the header
        ``TypeSafe-Danger-Zone-Images: true``; a text-only model answers 400 ``images_not_supported``."""
        body: dict = {"state": state, "questions": dict(questions)}
        if model:
            body["model"] = model
        if assets:
            body["assets"] = dict(assets)
        if self._adapter is not None:
            from .schema import ApiError
            kw = {"image_header": True} if assets and _takes(self._adapter.systemone, "image_header") else {}
            try:
                return self._adapter.systemone(body, **kw)
            except ApiError as e:
                raise JevStyleError(e.status, e.code, e.message, e.question, e.body()) from None
        headers = self._headers()
        if assets:
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        resp = self._client().post("/v1/systemone", json=body, headers=headers)
        if resp.status_code != 200:
            self._raise(resp)
        return resp.json()

    def models(self) -> dict:
        if self._adapter is not None:
            return self._adapter.models()
        resp = self._client().get("/v1/models", headers=self._headers())
        if resp.status_code != 200:
            self._raise(resp)
        return resp.json()

    def close(self) -> None:
        if self._http is not None and hasattr(self._http, "close"):
            self._http.close()

    def __enter__(self) -> "JevStyle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _takes(fn: Any, name: str) -> bool:
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _host_of(url: str | None) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url or "").hostname or "").lower()
    except ValueError:
        return ""


def is_loopback(url: str | None) -> bool:
    """``url`` points at this machine (127.0.0.0/8, localhost, ::1)."""
    host = _host_of(url)
    return host in LOOPBACK_HOSTS or host.startswith("127.")


def implicit_api_key(base_url: str | None) -> str | None:
    """``$JEV_STYLE_API_KEY`` when it may be sent to ``base_url`` without being named explicitly: in-process
    (never sent), a loopback server, or the server ``$JEV_STYLE_URL`` names. For any other URL the caller passes
    ``api_key=`` (``--key NAME=ENV`` for ``jev-style eval``, ``api_key_env`` for a cascade tier)."""
    key = os.environ.get("JEV_STYLE_API_KEY")
    if not key or base_url is None or is_loopback(base_url):
        return key
    configured = (os.environ.get("JEV_STYLE_URL") or "").rstrip("/")
    if configured and configured.lower() == base_url.rstrip("/").lower():
        return key
    return None


# -- engine targets ---------------------------------------------------------------------------------
TARGETS = ("http(s)://host:port, local[:release][:backend], hf:<repo id>, jevk5:<repo id or folder>, fake or "
           "cascade:<name or path>")


def parse_target(target: str) -> tuple[str, dict]:
    """An engine target -> (kind, JevStyle keyword arguments), without loading anything. Kinds: ``http``
    (``http(s)://...``), ``fake``, ``hf`` (``hf:<repo id>``), ``local`` (``local[:release][:backend]``, e.g.
    ``local:2b:mlx``), ``jevk5`` (``jevk5:<Hub repo id or local folder>``: JevK5 in-process, answering as
    ``jevk5-serve`` does, see ``jev_style.jevk5_engine``) and ``cascade`` (``cascade:<name or path to
    cascade.json>``)."""
    from .models import BACKENDS
    if target.startswith(("http://", "https://")):
        return "http", {"base_url": target}
    if target == "fake":
        return "fake", {"fake": True}
    for prefix, kind, key in (("hf:", "hf", "repo"), ("jevk5:", "jevk5", "jevk5"),
                              ("cascade:", "cascade", "cascade")):
        if target.startswith(prefix):
            if not target[len(prefix):]:
                raise ValueError(f"engine target {target!r}: nothing after {prefix!r}")
            return kind, {key: target[len(prefix):]}
    if target.split(":", 1)[0] == "local":
        kw: dict = {}
        for part in target.split(":")[1:]:
            if part in BACKENDS:
                kw["backend"] = part
            elif part:
                kw["release"] = part
            else:
                raise ValueError(f"engine target {target!r}: empty part")
        return "local", kw
    raise ValueError(f"unknown engine target {target!r} (expected {TARGETS})")


def from_target(target: str, *, backend: str = "auto", api_key: str | None = None, timeout: float | None = None,
                **load_kw: Any) -> JevStyle:
    """A client for an engine target (see ``parse_target``); used by ``jev-style eval --server`` and the cascade
    tiers. ``backend`` is the default for in-process targets that do not name one; ``api_key`` / ``timeout`` apply
    to http targets, ``load_kw`` (device, dtype, revision, ...) to in-process ones (jevk5: only ``revision`` and
    ``verify``)."""
    kind, kw = parse_target(target)
    if kind == "http":
        return JevStyle(base_url=kw["base_url"], api_key=api_key, **({"timeout": timeout} if timeout else {}))
    if kind == "fake":
        return JevStyle(fake=True)
    if kind == "hf":
        return JevStyle.from_pretrained(kw["repo"], **load_kw)      # the repo picks the backend
    if kind == "jevk5":
        from .jevk5_engine import load_jevk5
        return JevStyle(adapter=load_jevk5(kw["jevk5"], revision=load_kw.get("revision"),
                                           verify=load_kw.get("verify")))
    if kind == "cascade":
        return JevStyle(cascade=kw["cascade"], backend=backend, **load_kw)
    return JevStyle(**{"backend": backend, **load_kw, **kw})


# -- module-level shortcuts ------------------------------------------------------------------------
_default: JevStyle | None = None
_default_lock = threading.Lock()


def configure(**kw: Any) -> JevStyle:
    """Set the client the shortcuts use (same arguments as ``JevStyle``). Returns it."""
    global _default
    client = JevStyle(**kw)
    with _default_lock:
        old, _default = _default, client
    if old is not None:
        old.close()
    return client


def default_client() -> JevStyle:
    global _default
    with _default_lock:
        if _default is None:
            url = os.environ.get("JEV_STYLE_URL")
            backend = os.environ.get("JEV_STYLE_BACKEND", "auto")
            _default = JevStyle(base_url=url) if url else JevStyle(backend=backend,
                                                                   release=os.environ.get("JEV_STYLE_RELEASE"))
        return _default


def decide(state: Any, questions: Mapping[str, dict], model: str | None = None,
           assets: Mapping[str, Any] | None = None) -> dict:
    """``JevStyle.decide`` on the shared client."""
    return default_client().decide(state, questions, model=model, assets=assets)


def classify(state: Any, labels: Mapping[str, Any] | Iterable[str],
             instructions: Any = "Which label fits best?") -> dict:
    """Pick one label. ``labels`` is a list of names or {name: description}.
    Returns {"label", "confidence", "probabilities"}."""
    ans = decide(state, {"label": choice(instructions, labels)})["answers"]["label"]
    return {"label": ans["choice"], "confidence": ans["confidence"], "probabilities": ans["probabilities"]}
