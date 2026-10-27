"""Image requests (TypeSafe's ``assets`` / multipart, Jev 1.13 / 1.14) on a fake image runtime (no model needed)."""
import base64
import io
import json
import types

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from jev_style.adapter import Adapter
from jev_style.server import create_app

HDR = {"TypeSafe-Danger-Zone-Images": "true"}


def png(w=320, h=300, color=(200, 30, 30), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format=fmt)
    return buf.getvalue()


def b64(data):
    return base64.b64encode(data).decode()


class _BudgetError(ValueError):
    pass


class _QuestionError(ValueError):
    pass


class ImageRuntime:
    """decide_many(..., images=) like the 2B v4 runtime; records what it was given."""

    def __init__(self):
        self.calls = []

    def decide_many(self, state, questions, category=None, temperature=None, head_max=None, max_len=None, images=None):
        self.calls.append({"state": state, "images": list(images or [])})
        if images and any(not x.startswith(b"\x89PNG") and not x.startswith(b"\xff\xd8") for x in images):
            raise OSError("cannot identify image file")
        out = []
        for q in questions:
            names = ["false", "true"] if q["t"] == "noul" else (list(q["crit"]) if q["t"] == "choice"
                                                                 else [str(i) for i in range(len(q["crit"]))])
            out.append({"probabilities": {n: 1 / len(names) for n in names}, "input_tokens": 50, "head_tokens": 10})
        return out


class TextRuntime(ImageRuntime):
    def decide_many(self, state, questions, category=None):            # no images= (v3 runtimes)
        return super().decide_many(state, questions, category)


RT = types.SimpleNamespace(InputBudgetError=_BudgetError, QuestionError=_QuestionError)


def make(runtime):
    return TestClient(create_app(Adapter(runtime, RT, "torch", model_id="jev-style-2b-decision-v4"))), runtime


def body(state, assets, instructions="Is the door open in `photo`?"):
    return {"state": state, "assets": assets,
            "questions": {"open": {"type": "noul", "instructions": instructions}}}


def test_json_assets_become_state_placeholders():
    client, rt = make(ImageRuntime())
    img = png()
    r = client.post("/v1/systemone", headers=HDR,
                    json=body({"photo": "door-cam", "note": "night"}, {"door-cam": {"type": "image", "format": "png",
                                                                                 "data_base64": b64(img)}}))
    assert r.status_code == 200, r.text
    assert 0 <= r.json()["answers"]["open"]["noul"] <= 1
    call = rt.calls[-1]
    assert call["state"] == {"photo": "<|image_1|>", "note": "night"} and call["images"] == [img]


def test_every_use_adds_the_image_again_and_keys_are_not_replaced():
    client, rt = make(ImageRuntime())
    a, b = png(color=(1, 2, 3)), png(color=(9, 9, 9))
    assets = {"cam": {"format": "png", "data_base64": b64(a)}, "cam-2": {"format": "png", "data_base64": b64(b)}}
    state = {"cam": "before cam-2 and cam, then cam", "x": ["cam"]}
    r = client.post("/v1/systemone", headers=HDR, json=body(state, assets, "Did anything change?"))
    assert r.status_code == 200, r.text
    call = rt.calls[-1]
    assert call["state"] == {"cam": "before <|image_1|> and <|image_2|>, then <|image_3|>", "x": ["<|image_4|>"]}
    assert call["images"] == [b, a, a, a]                 # longest key first where both start; left to right


def test_multipart_parts():
    client, rt = make(ImageRuntime())
    img = png(fmt="JPEG")
    req = body({"photo": "door-cam"}, {"door-cam": {"type": "image", "part": "door-photo"}})
    r = client.post("/v1/systemone", headers=HDR,
                    files={"request": (None, json.dumps(req), "application/json"),
                           "door-photo": ("door.jpg", img, "image/jpeg")})
    assert r.status_code == 200, r.text
    assert rt.calls[-1]["images"] == [img]


@pytest.mark.parametrize("mutate,status,code", [
    (lambda b, h: h.clear(), 400, "images_header_required"),
    (lambda b, h: b["assets"]["door-cam"].update(data_base64="not base64!"), 400, "invalid_image"),
    (lambda b, h: b["assets"]["door-cam"].update(format="PNG"), 400, "invalid_image"),
    (lambda b, h: b["assets"]["door-cam"].update(format="jpg"), 400, "invalid_image"),          # data is a PNG
    (lambda b, h: b["assets"]["door-cam"].update(data_base64=b64(png(100, 100))), 400, "invalid_image"),
    (lambda b, h: b["assets"]["door-cam"].update(data_base64=b64(png(1600, 400))), 413, "image_too_large"),
    (lambda b, h: b["assets"].update(spare={"format": "png", "data_base64": b64(png())}), 400, "invalid_image"),
    (lambda b, h: b["assets"]["door-cam"].update(part="p"), 400, "invalid_image"),               # part AND data
    (lambda b, h: b["questions"]["open"].update(instructions="Is door-cam dark?"), 400, "images_not_supported"),
])
def test_rules(mutate, status, code):
    client, _ = make(ImageRuntime())
    b, h = body({"photo": "door-cam"}, {"door-cam": {"format": "png", "data_base64": b64(png())}}), dict(HDR)
    mutate(b, h)
    r = client.post("/v1/systemone", headers=h, json=b)
    assert (r.status_code, r.json()["error"]["code"]) == (status, code), r.text


def test_text_models_still_refuse_images():
    client, _ = make(TextRuntime())
    r = client.post("/v1/systemone", headers=HDR,
                    json=body({"photo": "door-cam"}, {"door-cam": {"format": "png", "data_base64": b64(png())}}))
    assert (r.status_code, r.json()["error"]["code"]) == (400, "images_not_supported")
    r = client.post("/v1/systemone", headers=HDR, files={"request": (None, "{}", "application/json")})
    assert (r.status_code, r.json()["error"]["code"]) == (400, "images_not_supported")
    assert client.get("/v1/models").json()["data"][0]["images"] is False


def test_models_reports_images_and_text_requests_still_work():
    client, rt = make(ImageRuntime())
    assert client.get("/v1/models").json()["data"][0]["images"] is True
    r = client.post("/v1/systemone", json={"state": "plain", "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    assert r.status_code == 200 and rt.calls[-1]["images"] == []


def test_client_sends_images_in_process_and_over_http(tmp_path):
    from jev_style import JevStyle, JevStyleError, image, noul
    img = png()
    f = tmp_path / "door.png"
    f.write_bytes(img)
    assert image(f) == {"type": "image", "format": "png", "data_base64": b64(img)}
    assert image(png(fmt="JPEG"))["format"] == "jpg"
    with pytest.raises(ValueError, match="PNG or JPEG"):
        image(b"GIF89a....")
    qs = {"open": noul("Is the door open?")}

    rt = ImageRuntime()                                               # in-process: the adapter gets image_header=True
    js = JevStyle(adapter=Adapter(rt, RT, "torch", model_id="jev-style-2b-decision-v4"))
    assert 0 <= js.decide({"photo": "cam"}, qs, assets={"cam": image(f)})["answers"]["open"]["noul"] <= 1
    assert rt.calls[-1] == {"state": {"photo": "<|image_1|>"}, "images": [img]}

    client, rt = make(ImageRuntime())                                 # over HTTP: the header goes with the body
    js = JevStyle(base_url="http://testserver", http_client=client)
    js.decide({"photo": "cam"}, qs, assets={"cam": image(img)})
    assert rt.calls[-1]["images"] == [img]

    client, _ = make(TextRuntime())                                   # a text-only model refuses them clearly
    with pytest.raises(JevStyleError) as e:
        JevStyle(base_url="http://testserver", http_client=client).decide({"photo": "cam"}, qs,
                                                                          assets={"cam": image(img)})
    assert (e.value.status, e.value.code) == (400, "images_not_supported")
