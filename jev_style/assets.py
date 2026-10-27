"""Images in ``/v1/systemone`` requests, the way TypeSafe's API takes them (``assets``, Jev 1.13 / 1.14).

Request shapes::

    JSON       {"state": {"photo": "door-cam"}, "assets": {"door-cam": {"type": "image", "format": "jpg",
                                                                    "data_base64": "..."}}, "questions": {...}}
    multipart  part "request" (application/json) = that JSON body, with {"part": "<file part name>"} in place of
               "format" + "data_base64"; one file part per image (image/png, image/jpeg or application/octet-stream)

Both need the header ``TypeSafe-Danger-Zone-Images: true``. A reference is an asset key matched as plain text inside
a string value: a value equal to the key is the image; in a longer string the keys are replaced from left to right
(the longest key first where several start at the same place); object keys are never replaced; every use adds the
image again. This model reads images as part of the shared state, so references belong in ``state``: a reference
in a question's instructions or criteria is refused with 400 ``images_not_supported`` (nothing is moved silently).
Each reference becomes ``<|image_k|>`` (k = 1..n in reading order) and the runtime gets the n images in that order.

Limits (TypeSafe's): PNG or JPEG still images; 16 MiB per image; 280..1568 px per side and 78,400..2,458,624 px in
total; all images together and the request with every reference expanded at most 32 MiB; reference keys at most
1,024 bytes each and 65,536 bytes together. Smaller than allowed or malformed -> 400, larger -> 413.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
from email.parser import BytesParser
from email.policy import default as email_default
from typing import Any

from .schema import MAX_BODY_BYTES, ApiError

HEADER = "typesafe-danger-zone-images"
FORMATS = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG"}
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 280, 1568
MIN_PIXELS, MAX_PIXELS = 280 * 280, 1568 * 1568
MAX_ALL_IMAGES = 32 * 1024 * 1024
MAX_EXPANDED = 32 * 1024 * 1024
MAX_KEY_BYTES, MAX_KEYS_BYTES = 1024, 65536
PART_TYPES = ("image/png", "image/jpeg", "image/jpg", "application/octet-stream")


def _bad(message: str, status: int = 400, code: str = "invalid_image", question: str | None = None) -> ApiError:
    return ApiError(status, code, message, question)


def header_on(headers: Any) -> bool:
    return str((headers or {}).get(HEADER, "")).strip().lower() == "true"


def split_multipart(raw: bytes, content_type: str) -> tuple[Any, dict[str, tuple[bytes, str]]]:
    """multipart/form-data body -> (the parsed "request" JSON part, {part name: (bytes, content type)})."""
    from .schema import parse_json
    msg = BytesParser(policy=email_default).parsebytes(b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + raw)
    if not msg.is_multipart():
        raise _bad("multipart body could not be parsed (check the boundary)", code="invalid_multipart")
    parts: dict[str, tuple[bytes, str]] = {}
    request = None
    for i, part in enumerate(msg.iter_parts()):
        name = part.get_param("name", header="content-disposition")
        if not name:
            raise _bad("every multipart part needs a name", code="invalid_multipart")
        data = part.get_payload(decode=True) or b""
        if name == "request":
            if i != 0:
                raise _bad('the JSON part "request" must come first', code="invalid_multipart")
            request = parse_json(data)
            continue
        if name in parts:
            raise _bad(f"duplicate multipart part {name!r}", code="invalid_multipart")
        ctype = part.get_content_type().lower()
        if ctype not in PART_TYPES:
            raise _bad(f"part {name!r} has content type {ctype}; use image/png, image/jpeg or application/octet-stream",
                       code="invalid_multipart")
        parts[name] = (data, ctype)
    if request is None:
        raise _bad('multipart requests need a first part named "request" (application/json)', code="invalid_multipart")
    return request, parts


def _check_image(key: str, data: bytes, fmt: str | None) -> None:
    if not data:
        raise _bad(f"asset {key!r}: empty image")
    if len(data) > MAX_IMAGE_BYTES:
        raise _bad(f"asset {key!r}: {len(data) / 2**20:.1f} MiB is over the 16 MiB per-image limit", 413, "image_too_large")
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - the torch extra installs Pillow
        raise _bad("image requests need Pillow (pip install \"jev-style[torch]\")", 400, "images_not_supported") from None
    try:
        im = Image.open(io.BytesIO(data))
        kind = im.format
        w, h = im.size
        animated = getattr(im, "is_animated", False)
        im.verify()
    except Exception as e:  # noqa: BLE001 - any decoder failure is a damaged / unsupported file
        raise _bad(f"asset {key!r}: not a readable PNG or JPEG image ({type(e).__name__})") from None
    if kind not in ("PNG", "JPEG") or animated:
        raise _bad(f"asset {key!r}: only still PNG or JPEG images are supported (got {kind}{', animated' if animated else ''})")
    if fmt is not None and FORMATS[fmt] != kind:
        raise _bad(f"asset {key!r}: format {fmt!r} does not match the image data ({kind})")
    if min(w, h) < MIN_SIDE or w * h < MIN_PIXELS:
        raise _bad(f"asset {key!r}: {w}x{h} px is below the minimum width, height or total pixels "
                   f"({MIN_SIDE} px per side, {MIN_PIXELS:,} px)")
    if max(w, h) > MAX_SIDE or w * h > MAX_PIXELS:
        raise _bad(f"asset {key!r}: {w}x{h} px is over the limit ({MAX_SIDE} px per side, {MAX_PIXELS:,} px)",
                   413, "image_too_large")


def load_assets(assets: Any, parts: dict[str, tuple[bytes, str]] | None) -> dict[str, bytes]:
    """Validate ``assets`` ({key: spec}) and return {key: image bytes}."""
    if not isinstance(assets, dict) or not assets:
        raise _bad("assets must be a non-empty object mapping reference -> image", code="invalid_request")
    parts = parts or {}
    total_keys, out, used_parts = 0, {}, set()
    for key, spec in assets.items():
        if not key:
            raise _bad("asset references must not be empty")
        nb = len(key.encode("utf-8"))
        total_keys += nb
        if nb > MAX_KEY_BYTES:
            raise _bad(f"asset reference {key[:40]!r}... is {nb} bytes; the limit is {MAX_KEY_BYTES}", 413, "reference_too_long")
        if not isinstance(spec, dict):
            raise _bad(f"asset {key!r} must be an object")
        if spec.get("type", "image") != "image":
            raise _bad(f"asset {key!r}: type must be \"image\"")
        has_part, has_data = "part" in spec, "data_base64" in spec
        if has_part == has_data:
            raise _bad(f"asset {key!r}: give exactly one of \"part\" (multipart) and \"data_base64\"")
        fmt = spec.get("format")
        if has_data:
            if fmt not in FORMATS:
                raise _bad(f"asset {key!r}: format must be one of png, jpg, jpeg (lowercase) with data_base64")
            if not isinstance(spec["data_base64"], str):
                raise _bad(f"asset {key!r}: data_base64 must be a string")
            try:
                data = base64.b64decode(spec["data_base64"], validate=True)
            except (binascii.Error, ValueError):
                raise _bad(f"asset {key!r}: data_base64 is not standard base64 (with = padding, no line breaks)") from None
        else:
            name = spec["part"]
            if fmt is not None and fmt not in FORMATS:
                raise _bad(f"asset {key!r}: format must be one of png, jpg, jpeg (lowercase)")
            if not isinstance(name, str) or name not in parts:
                raise _bad(f"asset {key!r}: multipart part {name!r} is missing", code="invalid_multipart")
            used_parts.add(name)
            data = parts[name][0]
        _check_image(key, data, fmt)
        out[key] = data
    if total_keys > MAX_KEYS_BYTES:
        raise _bad(f"asset references total {total_keys} bytes; the limit is {MAX_KEYS_BYTES}", 413, "reference_too_long")
    extra = set(parts) - used_parts
    if extra:
        raise _bad(f"multipart parts {sorted(extra)} are not used by any asset", code="invalid_multipart")
    return out


def _find(text: str, keys: list[str]) -> list[tuple[int, str]]:
    """(position, key) of every reference in ``text``, left to right, longest key first at a position."""
    hits, i = [], 0
    while i < len(text):
        best = None
        for k in keys:                                   # keys sorted longest first
            if text.startswith(k, i):
                best = k
                break
        if best is None:
            i += 1
            continue
        hits.append((i, best))
        i += len(best)
    return hits


def substitute(value: Any, images: dict[str, bytes], order: list[bytes], used: set[str]) -> Any:
    """``value`` with every reference replaced by ``<|image_k|>`` (appending the image to ``order``)."""
    keys = sorted(images, key=len, reverse=True)

    def walk(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}           # object keys are never replaced
        if isinstance(v, list):
            return [walk(x) for x in v]
        if not isinstance(v, str):
            return v
        if v in images:                                        # a whole value first
            order.append(images[v])
            used.add(v)
            return f"<|image_{len(order)}|>"
        hits = _find(v, keys)
        if not hits:
            return v
        out, last = [], 0
        for pos, k in hits:
            order.append(images[k])
            used.add(k)
            out.append(v[last:pos] + f"<|image_{len(order)}|>")
            last = pos + len(k)
        return "".join(out) + v[last:]

    return walk(value)


def references_in(value: Any, images: dict[str, bytes]) -> list[str]:
    """Asset keys referenced anywhere in ``value`` (string values only)."""
    keys = sorted(images, key=len, reverse=True)
    found: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            found.extend(k for _, k in ([(0, v)] if v in images else _find(v, keys)))

    walk(value)
    return found


def resolve(body: dict, parts: dict[str, tuple[bytes, str]] | None, header: bool) -> tuple[Any, list[bytes]]:
    """(state with ``<|image_k|>`` placeholders, the images in placeholder order) for a request with ``assets``."""
    if not header:
        raise _bad("requests with images need the header TypeSafe-Danger-Zone-Images: true", code="images_header_required")
    images = load_assets(body.get("assets"), parts)
    if sum(len(b) for b in images.values()) > MAX_ALL_IMAGES:
        raise _bad("all images together are over 32 MiB", 413, "payload_too_large")
    for qid, q in (body.get("questions") or {}).items():
        if isinstance(q, dict):
            refs = references_in({k: q.get(k) for k in ("instructions", "criteria")}, images)
            if refs:
                raise ApiError(400, "images_not_supported", f"image reference(s) {sorted(set(refs))} in this question: this "
                               "model reads images as part of the shared state only; put the reference in state", qid)
    order: list[bytes] = []
    used: set[str] = set()
    state = substitute(body.get("state"), images, order, used)
    unused = set(images) - used
    if unused:
        raise _bad(f"asset(s) {sorted(unused)} are not referenced in the request")
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    expanded = len(text.encode("utf-8")) + sum(len(b) for b in order)
    if expanded > MAX_EXPANDED or expanded > MAX_BODY_BYTES:
        raise _bad("the request with every image reference expanded is over 32 MiB", 413, "payload_too_large")
    return state, order
