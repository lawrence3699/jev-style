"""systemone-compatible requests on top of a Jev-Style runtime (``decide_many(state, questions)``).

One request = one state and any number of questions; they go to ONE ``decide_many`` call, so the
state is read once. Answer arithmetic (over the runtime's calibrated probabilities):

* noul:   ``noul`` = P(true)
* choice: ``choice`` = argmax option, ``probabilities`` = {option: p}, ``confidence``
* score:  ``score`` = sum_i i * p_i over level indices 0..K-1, ``legend`` = {"0": label, ...},
          ``probabilities`` = {"0": p_0, ...}, ``confidence``

Choice confidence = (k * p_max - 1) / (k - 1) for k options, clipped to [0, 1] (0 = uniform, 1 = all mass on
one option). Score confidence is TypeSafe's ordinal formula, which also counts how far probability sits from
the most likely level m: max(0, 1 - sum_i p_i |i - m| / MAD_unif), MAD_unif = (1/n) sum_i |i - (n - 1) / 2|.
Noul answers carry no confidence. Over-long inputs are rejected with 400 ``max_tokens_exceeded``, as in
TypeSafe's API; nothing is truncated.
"""
from __future__ import annotations

import inspect
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .models import MODEL_ID, RELEASES
from .schema import ApiError, ParsedRequest, QuestionSpec, parse_request

DEFAULT_CATEGORY = "typed_official"      # calibration group the model card uses for free-form typed questions


def confidence(probs: list[float]) -> float:
    k = len(probs)
    if k <= 1:
        return 1.0
    return float(min(1.0, max(0.0, (k * max(probs) - 1.0) / (k - 1.0))))


def score_confidence(probs: list[float]) -> float:
    """TypeSafe's ordinal Score confidence (docs.typesafe.ai/confidence); m = the first most likely level."""
    n = len(probs)
    if n <= 1:
        return 1.0
    m = max(range(n), key=probs.__getitem__)
    spread = sum(p * abs(i - m) for i, p in enumerate(probs))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return float(min(1.0, max(0.0, 1.0 - spread / even_spread)))


def renormalise(probs: list[float]) -> list[float]:
    clean = [float(p) if math.isfinite(p) and p > 0 else 0.0 for p in probs]
    s = sum(clean)
    if s <= 0:
        return [1.0 / len(clean)] * len(clean)
    return [p / s for p in clean]


def model_thread() -> ThreadPoolExecutor:
    """One worker thread that loads the model and runs every call. MLX keeps its GPU stream per thread,
    so the model must be used on the thread that loaded it; one worker also means one call at a time."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-style-model")


class Adapter:
    """Thread-safe: every model call runs on ``worker`` (one at a time). ``runtime`` is a Jev-Style runtime
    object, ``rt`` its module."""

    def __init__(self, runtime: Any, rt: Any, backend: str, model_id: str = MODEL_ID,
                 category: str | None = DEFAULT_CATEGORY, description: str | None = None,
                 worker: ThreadPoolExecutor | None = None, release_date: str | None = None):
        self.runtime, self.rt = runtime, rt
        self.worker = worker or model_thread()
        self.model_id = model_id
        self.release_date = release_date
        self.category = category
        self._backend = backend
        self.description = description or "Jev-Style 0.8B Decision v3: local typed decisions (noul / choice / score)"
        self._lock = threading.Lock()

    @property
    def backend(self) -> str:
        dev = getattr(self.runtime, "device", None)
        return f"{self._backend}-{dev}" if dev else self._backend

    @property
    def supports_images(self) -> bool:
        """The runtime reads images: its ``decide_many`` takes ``images=`` (2B v4 and later)."""
        fn = getattr(self.runtime, "decide_many", None)
        try:
            return fn is not None and "images" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

    @property
    def max_len(self) -> int:
        return int(getattr(getattr(self.runtime, "renderer", None), "max_len", 25_600))

    @property
    def head_max(self) -> int:
        return int(getattr(getattr(self.runtime, "renderer", None), "head_max", 2_048))

    def systemone(self, body: Any, *, parts: dict | None = None, image_header: bool = False) -> dict:
        """``parts`` / ``image_header``: a multipart request's image parts and whether it sent
        ``TypeSafe-Danger-Zone-Images: true`` (see ``assets``); images reach the runtime as ``images=``."""
        t0 = time.perf_counter()
        parsed = body if isinstance(body, ParsedRequest) else parse_request(
            body, images_ok=self.supports_images, parts=parts, image_header=image_header)
        budget = getattr(self.rt, "InputBudgetError", ())
        qerr = getattr(self.rt, "QuestionError", ())
        kw = {"category": self.category}
        if parsed.images:
            kw["images"] = parsed.images
        with self._lock:
            try:
                results = self.worker.submit(self.runtime.decide_many, parsed.state,
                                             [q.internal for q in parsed.questions], **kw).result()
            except budget as e:
                what = "images and text together exceed the context length: " if parsed.images else ""
                raise ApiError(400, "max_tokens_exceeded", f"{what}{e} (nothing truncated)") from None
            except qerr as e:
                raise ApiError(422, "invalid_question", str(e)) from None
            except (OSError, ValueError) as e:
                if not parsed.images:
                    raise
                raise ApiError(400, "invalid_image", f"the model could not read an image: {e}") from None
        total_ms = (time.perf_counter() - t0) * 1000
        answers = {q.id: self._answer(q, r) for q, r in zip(parsed.questions, results)}
        head = sum(int(r.get("head_tokens", 0)) for r in results)
        state_tokens = int(results[0]["input_tokens"]) - int(results[0].get("head_tokens", 0)) if results else 0
        return {"model": self.model_id, "answers": answers,
                "usage": {"input_tokens": state_tokens + head, "state_tokens": state_tokens, "output_tokens": 0},
                "latency_ms": round(total_ms, 1), "timing": {"total_ms": round(total_ms, 1)}, "backend": self.backend}

    @staticmethod
    def _answer(q: QuestionSpec, res: dict) -> dict:
        probs = res["probabilities"]
        names = q.options if q.type != "noul" else ["false", "true"]
        p = renormalise([float(probs[n]) for n in names])
        if q.type == "noul":
            return {"type": "noul", "noul": p[1]}
        i = max(range(len(p)), key=p.__getitem__)
        if q.type == "choice":
            return {"type": "choice", "choice": names[i], "confidence": confidence(p),
                    "probabilities": dict(zip(names, p))}
        return {"type": "score", "score": float(sum(k * v for k, v in enumerate(p))),
                "confidence": score_confidence(p), "legend": dict(q.legend), "probabilities": dict(zip(names, p))}

    def models(self) -> dict:
        entry = {"id": self.model_id, "context_tokens": self.max_len, "head_max_tokens": self.head_max,
                 "backend": self.backend, "images": self.supports_images, "box": False}
        return {"object": "list", "data": [entry],
                # the shape the public systemone SDKs read from GET /v1/models; they require release_date to be a
                # string ("" for the fake engine / unpinned builds)
                "models": [{"name": self.model_id, "release_date": self.release_date or "",
                            "description": self.description}]}

    def release_info(self) -> dict:
        fake = self.model_id.endswith("fake")
        return {"model_name": self.model_id, "release_kind": "fake" if fake else "release", "untrained": fake,
                "acceptance_status": None}


def build_adapter(backend: str = "auto", *, fake: bool = False, **load_kw: Any) -> Adapter:
    if fake:
        from . import fake as fk
        return Adapter(fk.FakeRuntime(), fk, "fake", fk.FAKE_MODEL_ID,
                       description="Fake engine: deterministic hash-based probabilities, no model")
    from .models import load_release
    worker = model_thread()
    runtime, rt, build = worker.submit(load_release, backend, **load_kw).result()
    if build.extra.get("prefix_sharing"):       # only where it was shown to give the same scores
        from .fastpath import enable_prefix_sharing
        enable_prefix_sharing(runtime, rt)      # MLX: read the state once for many questions
    rel = RELEASES.get(build.release)            # None for a repo loaded with trust_remote_code
    return Adapter(runtime, rt, build.backend, rel.model_id if rel else build.repo, worker=worker,
                   release_date=rel.release_date if rel else None,
                   description=f"{rel.title if rel else build.repo}: local typed decisions (noul / choice / score)")
