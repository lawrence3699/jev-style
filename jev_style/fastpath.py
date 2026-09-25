"""Read-once, ask-many for the MLX runtime.

The MLX runtime shipped with the weights scores every question with its own pass over state +
question: it prefills whole ``PREFILL_CHUNK`` blocks of the input, then runs the rest (the tail of
the state plus the question) in one call. With many questions about one long state (context
compaction asks ~70 questions about a ~20K-token state) that re-reads the state every time.

``enable_prefix_sharing`` keeps that exact computation but shares the work: the whole blocks of the
state are prefilled once into a prompt cache, each question continues from a copy of that cache and
runs exactly the same remaining pieces as the original ``_scores`` would (same block boundaries,
same final call). So the scores are the ones the runtime computes on its own, while each question
only re-reads at most one block (< 2,048 tokens) of the state.
"""
from __future__ import annotations

import copy
from typing import Any


def _shared_cut(r: Any, chunk: int) -> int:
    """Tokens of r's input that ``_scores`` prefills in whole blocks and that belong to the state."""
    if len(r.ids) <= chunk:
        return 0
    cut = (min(r.slots) // chunk) * chunk
    return min(cut, (r.prefix_len // chunk) * chunk)


def enable_prefix_sharing(runtime: Any, rt: Any) -> bool:
    """Patch ``runtime._scores_many`` (MLX runtime only). Returns True if patched.

    Measured on an M1 Max (14 questions): identical probabilities (max difference 0.0) and 5.1x / 8.0x
    faster at 6.7K / 18.9K state tokens. Reading the whole state once and running only each question's
    own tokens is ~1.4x faster still but changes the scores (block boundaries move): up to 0.11 in
    probability and 3 of 14 top answers at 18.9K tokens, so it is not offered.
    """
    if getattr(runtime, "backend", None) != "mlx" or not hasattr(runtime, "inner"):
        return False
    chunk = int(getattr(rt, "PREFILL_CHUNK", 2048))
    mx = runtime.mx
    single = runtime._scores

    def prefill(cache: Any, ids: list[int], start: int, stop: int) -> None:
        for s in range(start, stop, chunk):
            runtime.inner(mx.array(ids[s:s + chunk])[None], cache=cache)
            mx.eval([c.state for c in cache])

    def scores_many(rendered: list) -> list:
        if len(rendered) < 2:
            return [single(r) for r in rendered]
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(rendered):
            groups.setdefault(tuple(r.ids[:r.prefix_len]), []).append(i)
        out: list = [None] * len(rendered)
        for idx in groups.values():
            shared = min(_shared_cut(rendered[i], chunk) for i in idx)
            if len(idx) < 2 or shared == 0:
                for i in idx:
                    out[i] = single(rendered[i])
                continue
            base = runtime._make_cache()
            prefill(base, rendered[idx[0]].ids, 0, shared)
            for i in idx:
                r = rendered[i]
                ids, slots = r.ids, r.slots
                cut = (min(slots) // chunk) * chunk if len(ids) > chunk else 0
                cache = copy.deepcopy(base)
                prefill(cache, ids, shared, cut)            # the same blocks _scores would prefill
                h = runtime.inner(mx.array(ids[cut:])[None], cache=cache)[0]
                v = h[mx.array([x - cut for x in slots])].astype(mx.float32) @ runtime.direction
                mx.eval(v)
                out[i] = v.tolist()
                del cache, h, v
            del base
        return out

    runtime._scores_many = scores_many
    return True
