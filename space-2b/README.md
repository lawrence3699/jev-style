---
title: Jev-Style · Local Agent Decisions
emoji: ⚖️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Route tickets, review agent commands and check receipts
models:
  - chaoliangUNSW/Jev-Style-2B-Decision-v4
  - chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF
  - chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX
preload_from_hub:
  - chaoliangUNSW/Jev-Style-2B-Decision-v4 LICENSE,NOTICE,SHA256SUMS.json,chat_template.jinja,config.json,generation_config.json,jev_style_decision.py,macjev_readout.json,manifest.json,model-00001-of-00003.safetensors,model-00002-of-00003.safetensors,model-00003-of-00003.safetensors,model.safetensors.index.json,preprocessor_config.json,readout_config.json,release_config.json,requirements.txt,tokenizer.json,tokenizer_config.json,video_preprocessor_config.json,src/macjev/__init__.py,src/macjev/backbone.py,src/macjev/eval/__init__.py,src/macjev/eval/aggregate.py,src/macjev/eval/di_v4.py,src/macjev/eval/metrics.py,src/macjev/eval/predictions.py,src/macjev/paths.py,src/macjev/readout.py,src/macjev/render.py,src/macjev/render_v2.py,src/macjev/runtime/__init__.py,src/macjev/runtime/cuda_graphs.py,src/macjev/runtime/fast_text.py,src/macjev/runtime/v4_engine.py,src/macjev/schema.py,src/macjev/serve/__init__.py,src/macjev/serve/schema.py,src/macjev/train/__init__.py,src/macjev/train/block_attention.py,src/macjev/train/data.py,src/macjev/train/evaluate.py,src/macjev/train/model.py,src/macjev/vision.py c9504148dae7f7a43d67a5782f12a015a4b4022c
tags:
  - text-classification
  - image-classification
  - llm-routing
  - guardrails
  - calibration
---

# Jev-Style 2B v4 — live decisions and saved examples

Route a support ticket, review an agent command, or check a receipt. The model returns a probability for each allowed option. Choose a task, edit its input, and select **Run live**.

[Project home](https://jevstyle.com/?utm_source=huggingface&utm_medium=space&utm_campaign=demo) · [Model and evaluations](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4) · [Python package and source](https://github.com/lawrence3699/jev-style)

## When a live request cannot finish

Live inference uses shared Hugging Face ZeroGPU resources. Queue capacity, account quota, timeouts, invalid inputs, and other runtime failures have different causes. The interface preserves your input, describes the known cause without exposing server exceptions, and leaves two routes available:

- **Saved examples:** three genuine, previously computed outputs with the exact recorded input, question, options, model revision, and cache provenance. These live in a separate labelled panel. They never replace the live output or claim to answer edited input. Expanding a saved card makes no backend or GPU request after the page has loaded.
- **Run on your machine:** copy the platform-specific setup command. The current request's Python export is prepared before GPU inference, so a quota failure does not prevent local reproduction. Save a selected image as `image.png` next to the script.

The wait indicator measures time spent waiting in the interface; it does not predict an ETA. **Stop waiting** cancels queued work and stops receiving the live result. Work already running on the GPU may finish in the background. The app does not retry automatically.

A platform rejection can happen before an application callback runs. Gradio's native queue/error UI remains enabled, and the already-loaded saved cards and local setup do not depend on that callback succeeding. If model initialization fails, the page can still serve saved examples.

## Local setup

Use Python 3.10 or newer in a virtual environment:

```bash
python -m pip install -U "jev-style[torch]>=0.6.1"
jev-style serve --release 2b-v4 --backend torch
```

Open `http://127.0.0.1:8765`. For Apple silicon **text-only** use `jev-style[mlx]` and `--backend mlx`. Images require PyTorch. First launch downloads the model; weight file size is not the same as RAM/VRAM usage.

The model reads text, JSON, and optionally an image. It supports `choice`, `noul` (yes/no), and `score`, with up to 32,768 input tokens (including image tokens), up to 255 choices, and at most 12 questions per request in this Space. Stock text-generation runtimes do not produce these decision scores.

## Saved result provenance

`saved_examples.json` contains three outputs recovered with read-only HTTP GETs from the previous Space's Gradio cache. Every entry includes the original probabilities, the exact input and question, source CSV hashes, and pinned model/Space revisions. The receipt image is checked against its recorded SHA-256 before embedding. Original cache timestamps do not encode a timezone; they are never labelled UTC. Historical timing is shown only inside recording details and is not a current request time.

Sample inputs are illustrative, not a benchmark or authorization policy. The agent-command prediction is not a safety guarantee. Evaluate on representative held-out data before automating any action.

The live model remains pinned to `chaoliangUNSW/Jev-Style-2B-Decision-v4@c9504148dae7f7a43d67a5782f12a015a4b4022c`, PyTorch float32. CUDA graphs remain disabled on ZeroGPU. No model weights or decision thresholds were changed for this interface update.

## Developer preview

With Gradio 6.28.0 installed, `JEV_DEMO_PREVIEW=1 python app.py` starts a clearly marked cache-only UI preview without downloading or loading a model. Do not set this variable on the hosted live Space. The public `decide` and `decide_many` endpoints retain their previous input shapes and three output fields; failures remain errors and are not substituted with saved outputs.

The longer document example uses public-domain U.S. founding documents from Project Gutenberg. The receipt is synthetic. Code is Apache-2.0. The model was trained in part on soft labels distilled from TypeSafe Jev 1.14; this is an independent project, not affiliated with TypeSafe or Jev.
