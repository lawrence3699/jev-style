"""Python client for Jev-Style: in-process, or against a running ``jev-style serve``.

    from jev_style import JevStyle, noul, choice, score
    js = JevStyle(fake=True)                             # in-process fake engine (no model)
    js = JevStyle(backend="auto")                        # in-process real model (downloads it once)
    js = JevStyle(base_url="http://127.0.0.1:8765")      # over HTTP
    out = js.decide("I was charged twice.", {
        "billing": noul("Is this about billing?"),
        "team": choice("Which team?", {"billing": None, "tech": "bugs, outages"}),
        "urgency": score("How urgent?", ["low", "medium", "high"]),
    })
    out["answers"]["team"]["choice"]

``decide`` returns exactly the HTTP response body of ``POST /v1/systemone`` and raises
``JevStyleError`` (status, code, message, question) for the contract's error bodies.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Mapping


class JevStyleError(Exception):
    def __init__(self, status: int, code: str, message: str, question: str | None = None, body: dict | None = None):
        super().__init__(f"{status} {code}: {message}" + (f" (question {question!r})" if question else ""))
        self.status, self.code, self.message, self.question, self.body = status, code, message, question, body or {}


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


class JevStyle:
    def __init__(self, base_url: str | None = None, backend: str = "auto", fake: bool = False,
                 api_key: str | None = None, timeout: float = 120.0, http_client: Any = None, **load_kw: Any):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key if api_key is not None else os.environ.get("JEV_STYLE_API_KEY")
        self.timeout = timeout
        self._http = http_client
        self._adapter = None
        if self.base_url is None and http_client is None:
            from .adapter import build_adapter
            self._adapter = build_adapter(backend, fake=fake, **load_kw)

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
        err = body.get("error") if isinstance(body, dict) else None
        err = err if isinstance(err, dict) else {}
        raise JevStyleError(resp.status_code, err.get("code", "http_error"), err.get("message", resp.text[:500]),
                          err.get("question"), body)

    # -- API ---------------------------------------------------------------------------------
    def decide(self, state: Any, questions: Mapping[str, dict], model: str | None = None) -> dict:
        body: dict = {"state": state, "questions": dict(questions)}
        if model:
            body["model"] = model
        if self._adapter is not None:
            from .schema import ApiError
            try:
                return self._adapter.systemone(body)
            except ApiError as e:
                raise JevStyleError(e.status, e.code, e.message, e.question, e.body()) from None
        resp = self._client().post("/v1/systemone", json=body, headers=self._headers())
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
