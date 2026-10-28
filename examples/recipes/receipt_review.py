"""Flag a receipt for review; does not approve expenses or move money."""
from __future__ import annotations

import argparse
from pathlib import Path

from jev_style import JevStyle, image, noul

try:
    from .common import emit, probability_arg
except ImportError:
    from common import emit, probability_arg

SAMPLE = Path(__file__).with_name("data") / "synthetic_receipt.png"
POLICY = "A meal expense with a total strictly greater than EUR 50 requires manager review."


def request_for(path):
    # The asset ID occurs only in state, not in the question's instructions.
    return {"state": {"receipt": "receipt_asset_1", "policy": POLICY},
            "questions": {"review": noul("Does this meal receipt require manager review under the policy?")},
            "assets": {"receipt_asset_1": image(path)}}


def review(js, path, minimum=0.8):
    models = js.models().get("data", [])
    if not models or not all(model.get("images") is True for model in models):
        raise ValueError("Use a 2B v4 PyTorch server: jev-style serve --release 2b --backend torch")
    response = js.decide(**request_for(path))
    probability = probability_arg(response["answers"]["review"]["noul"])
    return {"kind": "receipt_review", "model": response.get("model"),
            "manager_review_probability": probability,
            "recommendation": "manager_review" if probability >= minimum else "human_verification",
            "note": "A low score is not automatic expense approval. Verify the receipt and policy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--image", type=Path, default=SAMPLE)
    parser.add_argument("--min-probability", type=probability_arg, default=0.8)
    parser.add_argument("--check", action="store_true", help="validate image and build request without calling a model")
    args = parser.parse_args()
    if args.check:
        from jev_style.assets import load_assets
        request = request_for(args.image)
        decoded = load_assets(request["assets"], None)
        emit({"kind": "request_check", "image": str(args.image), "image_bytes": len(decoded["receipt_asset_1"]),
              "policy": POLICY, "inference_run": False})
        return
    with JevStyle(base_url=args.url, api_key="") as js:
        emit(review(js, args.image, args.min_probability))


if __name__ == "__main__":
    main()
