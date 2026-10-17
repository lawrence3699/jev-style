"""systemone request rules and compression (TypeSafe API, Jev 1.14 / SDK 0.7.x), on the fake engine."""
import gzip
import json

import pytest
import zstandard

from jev_style import schema, server
from jev_style.adapter import score_confidence

BODY = {"model": "jev-latest", "state": "Help! My payouts have been failing for 3 days. " * 40,
        "questions": {"urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                      "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                                      "criteria": ["Calm", "Frustrated", "Very angry"]}}}
RAW = json.dumps(BODY).encode()


def post(client, content, **headers):
    return client.post("/v1/systemone", content=content, headers={"content-type": "application/json", **headers})


def err(r):
    return r.status_code, r.json()["error"]["code"]


# ----------------------------------------------------------------------------- compression
@pytest.mark.parametrize("enc,data", [
    ("gzip", gzip.compress(RAW)),
    ("x-gzip", gzip.compress(RAW)),
    ("zstd", zstandard.ZstdCompressor().compress(RAW)),
    ("ZSTD", zstandard.ZstdCompressor().compress(RAW)),
    ("identity", RAW),
])
def test_compressed_bodies_answer_like_plain_ones(client, enc, data):
    plain = post(client, RAW).json()["answers"]
    r = post(client, data, **{"content-encoding": enc})
    assert r.status_code == 200 and r.json()["answers"] == plain


def test_zstd_streamed_frame_and_several_frames(client):
    plain = post(client, RAW).json()["answers"]
    streamed = zstandard.ZstdCompressor(write_content_size=False).compressobj()
    one = streamed.compress(RAW) + streamed.flush()                     # no content size in the frame header
    assert post(client, one, **{"content-encoding": "zstd"}).json()["answers"] == plain
    c = zstandard.ZstdCompressor()
    two = c.compress(RAW[:500]) + c.compress(RAW[500:])
    assert post(client, two, **{"content-encoding": "zstd"}).json()["answers"] == plain
    members = gzip.compress(RAW[:500]) + gzip.compress(RAW[500:])
    assert post(client, members, **{"content-encoding": "gzip"}).json()["answers"] == plain


@pytest.mark.parametrize("enc,data,expected", [
    ("br", RAW, (415, "unsupported_content_encoding")),
    ("gzip, zstd", RAW, (415, "unsupported_content_encoding")),
    ("gzip", RAW, (400, "invalid_compressed_body")),
    ("gzip", gzip.compress(RAW)[:-12], (400, "invalid_compressed_body")),
    ("zstd", RAW, (400, "invalid_compressed_body")),
    ("zstd", zstandard.ZstdCompressor().compress(RAW)[:-6], (400, "invalid_compressed_body")),
    ("zstd", b"", (400, "invalid_compressed_body")),
])
def test_bad_encodings(client, enc, data, expected):
    assert err(post(client, data, **{"content-encoding": enc})) == expected


def test_zstd_window_over_8_mib_is_refused():
    params = zstandard.ZstdCompressionParameters.from_level(3, window_log=24)      # 16 MiB window
    data = zstandard.ZstdCompressor(compression_params=params).compress(b"x" * (17 * 2**20))
    with pytest.raises(schema.ApiError) as e:
        schema.decode_body(data, "zstd")
    assert (e.value.status, e.value.code) == (400, "invalid_compressed_body")


@pytest.mark.parametrize("enc", ["gzip", "zstd"])
def test_limit_applies_to_the_decompressed_body(monkeypatch, enc):
    monkeypatch.setattr(schema, "MAX_BODY_BYTES", 4096)
    big = b"x" * 10_000
    data = gzip.compress(big) if enc == "gzip" else zstandard.ZstdCompressor().compress(big)
    assert len(data) < 4096
    with pytest.raises(schema.ApiError) as e:
        schema.decode_body(data, enc)
    assert (e.value.status, e.value.code) == (413, "payload_too_large")
    assert schema.decode_body(data[:0] + (gzip.compress(b"{}") if enc == "gzip" else
                                          zstandard.ZstdCompressor().compress(b"{}")), enc) == b"{}"


def test_raw_body_over_the_limit_is_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 1024)
    assert err(post(client, RAW)) == (413, "payload_too_large")


# ----------------------------------------------------------------------------- request rules
@pytest.mark.parametrize("raw,expected", [
    (b"\xef\xbb\xbf" + RAW, (400, "invalid_encoding")),
    (RAW.replace(b"Help!", b"Help\xff"), (400, "invalid_encoding")),
    (b'{"state": NaN, "questions": {}}', (400, "non_finite_number")),
    (b'{"state": [Infinity], "questions": {}}', (400, "non_finite_number")),
    (b'{"state": {"x": 1e400}, "questions": {}}', (400, "non_finite_number")),
    (b'{"state": [1' + b"0" * 400 + b'], "questions": {}}', (400, "non_finite_number")),
    (b'{"state": ' + b"[" * 5000 + b"]" * 5000 + b', "questions": {}}', (400, "nesting_too_deep")),
    (b'{"state": ' + b"[" * 200_000 + b"]" * 200_000 + b', "questions": {}}', (400, "nesting_too_deep")),
    (b"{nope", (422, "invalid_json")),
], ids=["bom", "not-utf8", "nan", "infinity", "1e400", "huge-int", "deep", "deeper-than-the-parser", "not-json"])
def test_request_rules(client, raw, expected):
    assert err(post(client, raw)) == expected


def nested(depth):
    v = "leaf"
    for _ in range(depth):
        v = {"k": v}
    return v


@pytest.mark.parametrize("where", ["state", "instructions", "criteria"])
def test_nesting_limit_is_64_levels(client, where):
    def body(depth):
        q = {"type": "noul", "instructions": "ok?"}
        b = {"state": "s", "questions": {"q": q}}
        if where == "state":
            b["state"] = nested(depth)
        elif where == "instructions":
            q["instructions"] = nested(depth)
        else:
            q.update(type="choice", criteria={"a": nested(depth - 1), "b": None})
        return b
    assert client.post("/v1/systemone", json=body(64)).status_code == 200
    r = client.post("/v1/systemone", json=body(65))
    assert err(r) == (400, "nesting_too_deep") and str(65) in r.json()["error"]["message"]


def test_nesting_counts_objects_and_arrays():
    assert [schema.nesting(v) for v in ("s", 1, {}, [], {"a": []}, [[{"b": 1}], 2])] == [0, 0, 1, 1, 2, 3]


# ----------------------------------------------------------------------------- text only
def test_images_and_box_are_refused(client):
    with_assets = {**BODY, "assets": {"img": {"data_base64": "AAAA", "format": "png"}}}
    assert err(client.post("/v1/systemone", json=with_assets)) == (400, "images_not_supported")
    r = client.post("/v1/systemone", content=b"--x\r\n", headers={"content-type": "multipart/form-data; boundary=x"})
    assert err(r) == (400, "images_not_supported")
    box = {"state": "s", "questions": {"where": {"type": "box", "instructions": "Find the logo"}}}
    r = client.post("/v1/systemone", json=box)
    assert err(r) == (400, "unsupported_question_type") and r.json()["error"]["question"] == "where"


@pytest.mark.parametrize("model", ["jev-latest", "jev-preview", "jev-1.14.0", "jev-1.13.0"])
def test_official_model_names_are_accepted(client, model):
    assert client.post("/v1/systemone", json={**BODY, "model": model}).status_code == 200


# ----------------------------------------------------------------------------- score confidence
@pytest.mark.parametrize("probs,expected", [
    ([0.0, 0.57, 0.43], 1 - 0.43 / (2 / 3)),          # docs.typesafe.ai/confidence, bug_severity example
    ([0.0, 0.5, 0.5], 0.25),                          # torn between neighbours
    ([0.5, 0.0, 0.5], 0.0),                           # torn between the ends
    ([0.0, 0.0, 0.48, 0.52], 0.52),
    ([0.0, 0.95, 0.05], 0.925),
    ([0.2] * 5, 0.0),
    ([0.0, 1.0], 1.0),
])
def test_score_confidence_matches_typesafe(probs, expected):
    assert score_confidence(probs) == pytest.approx(expected)


def test_score_answers_use_the_ordinal_confidence(client):
    a = client.post("/v1/systemone", json=BODY).json()["answers"]["frustration"]
    probs = [a["probabilities"][str(i)] for i in range(3)]
    assert a["confidence"] == pytest.approx(score_confidence(probs))


def test_model_list_release_date_is_always_a_string(client):
    for m in client.get("/v1/models").json()["models"]:
        assert isinstance(m["name"], str) and isinstance(m["description"], str) and isinstance(m["release_date"], str)
