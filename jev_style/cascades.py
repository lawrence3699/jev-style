"""Named cascade releases: frozen cascade configs that ship with the package (``--cascade NAME``).

Each entry is a cascade config dict (the ``cascade.json`` schema, see ``jev_style.cascade.CascadeConfig``). A named
cascade is frozen: its thresholds and ``frozen_utc`` are set once, from a calibration set whose sha256 is recorded,
and only change with a package release. One whose threshold is still a placeholder is listed but refused when
loaded (``jev_style.cascade.resolve_config``); its tiers can still be downloaded ahead of time.

* ``cascade-9b``  Jev-Style 2B Decision v3 (in-process, PyTorch float32, CUDA graphs on CUDA) -> JevK5-9B v0.3.3 by
  alibiserikbay (third party, Apache-2.0), in-process through the jevk5 package (``jev_style.jevk5_engine``). One
  threshold on the 2B's normalized confidence.
"""
from __future__ import annotations

import copy
from typing import Any

from .jevk5_engine import JEVK5_9B

# ---------------------------------------------------------------------------------------------------------------
# cascade-9b's frozen threshold tau and the time it was frozen: the lowest tau whose calibration accuracy stays within
# 1.0 point of JevK5-9B alone, with the 2B on its CUDA-graph path (rule fixed before any calibration result).
TAU_CASCADE_9B: float | None = 0.75                 # frozen 2026-10-02 on the calibration set below (see CHANGELOG)
FROZEN_UTC_CASCADE_9B: str | None = "2026-10-02T21:53:31Z"
# ---------------------------------------------------------------------------------------------------------------
CALIBRATION_SHA256_CASCADE_9B = "580962e4b2bc86b32b42731eda42536f57321c7d1ca9ac1e3b970bf7dbd7e2d2"

CASCADES: dict[str, dict[str, Any]] = {
    "cascade-9b": {
        "id": "jev-style-cascade-9b",
        "description": "Jev-Style 2B Decision v3 (in-process, PyTorch) -> JevK5-9B v0.3.3 by alibiserikbay (third "
                       "party, Apache-2.0, in-process via allebee/jevk5); one threshold on normalized confidence",
        "tiers": [
            {"name": "2b", "model_id": "jev-style-2b-decision-v3", "target": "local:2b-v3:torch", "dtype": "float32"},
            {"name": "jevk5-9b", "model_id": JEVK5_9B.repo, "target": "jevk5:" + JEVK5_9B.repo,
             "revision": JEVK5_9B.revision},
        ],
        "thresholds": [TAU_CASCADE_9B],
        "confidence": "normalized_pmax",
        "frozen_utc": FROZEN_UTC_CASCADE_9B,
        "calibration_sha256": CALIBRATION_SHA256_CASCADE_9B,
    },
}


def names() -> list[str]:
    return list(CASCADES)


def get(name: str) -> dict[str, Any]:
    """A copy of a named cascade's config, without the optional fields that are not set."""
    d = copy.deepcopy(CASCADES[name.strip().lower()])
    for k in ("frozen_utc", "calibration_sha256"):
        if d.get(k) is None:
            d.pop(k, None)
    return d


def placeholders(name: str) -> list[str]:
    """The fields of a named cascade that are still placeholders (a frozen cascade has none)."""
    d = CASCADES[name.strip().lower()]
    out = [f"thresholds[{i}]" for i, t in enumerate(d["thresholds"]) if t is None]
    if d.get("frozen_utc") is None:
        out.append("frozen_utc")
    return out
