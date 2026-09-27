"""Jev-Style: small, calibrated, local decision models with a systemone-compatible API."""
__version__ = "0.3.0"

from .client import JevStyle, JevStyleError, choice, classify, configure, decide, noul, score  # noqa: E402

__all__ = ["JevStyle", "JevStyleError", "choice", "classify", "configure", "decide", "noul", "score", "__version__"]
