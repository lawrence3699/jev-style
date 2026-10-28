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

# Jev-Style 2B v4

**Try a task:** route a support ticket, review an agent command or check a receipt. Choose an example, run a decision, then copy matching Python code beside the result.

[Install locally](https://github.com/lawrence3699/jev-style#quick-start) · [Agent integrations](https://github.com/lawrence3699/jev-style#agent-skills) · [Project home](https://jevstyle.com/?utm_source=huggingface&utm_medium=space&utm_campaign=demo)

Try [Jev-Style-2B-Decision-v4](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4): give it a text or a
JSON state, optionally with an image, and a typed question, get a calibrated probability for every option. Choice,
yes/no or score; up to 32,768 tokens of input (image tokens included) and up to 255 options. Ask several questions
about one state and the state is computed once.

Runs the model repo's own PyTorch runtime (float32) on ZeroGPU. Text-only builds:
[GGUF](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-GGUF) ·
[MLX](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4-MLX). The smaller model:
[Jev-Style v3 0.8B Space](https://huggingface.co/spaces/chaoliangUNSW/jev-style-v3). Website: [jevstyle.com](https://jevstyle.com).

**Quick local setup** (Python 3.10+):

```bash
pip install "jev-style[torch]>=0.6.0"
jev-style serve --release 2b-v4 --backend torch
```

Open `http://127.0.0.1:8765` for the local Playground. On Apple silicon, use `jev-style[mlx]` for text-only
requests; images require PyTorch. [Package documentation](https://github.com/lawrence3699/jev-style#quick-start).

**Direct-runtime Python** (Python with `torch`, `torchvision`, `transformers`, `tokenizers`, `numpy`, `pillow`; versions in
the model repo's `requirements.txt`):

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("chaoliangUNSW/Jev-Style-2B-Decision-v4")
sys.path.insert(0, path)
from jev_style_decision import JevStyleDecision

m = JevStyleDecision(path)                     # float32; CUDA, Apple MPS or CPU
r = m.decide("I was charged twice for my subscription this month.",
             "Which team should handle this ticket?",
             options={"billing": "payments, invoices, refunds", "technical": "bugs and outages",
                      "sales": "new purchases"})
print(r["answer"], r["probabilities"])

receipt = {"receipt": "<|image_1|>", "policy": "Meals over 50 EUR need a manager approval."}
r = m.decide(receipt, "Does this receipt need a manager approval?", images=[path + "/assets/receipt.png"])
print(r["answer"], r["probabilities"])
```

`m.decide_many(state, [q1, q2, ...])` scores several questions about one state; each question is
`{"t": "choice" | "noul" | "score", "ins": ..., "crit": ...}`. `images=[...]` takes paths, bytes, data URLs, base64 or
PIL images. The "Copy Python" box in the app shows direct-runtime code for whatever you ran.

Scope: the decision scores need the model's own runtimes (PyTorch here; the GGUF and MLX builds ship their own,
text only); stock llama.cpp, Ollama, LM Studio or `mlx_lm.generate` can load the weights but cannot produce them.
Inputs over 32,768 tokens are refused, never truncated. Images go in the state at native resolution up to 1.6 MP;
questions and options are text.

The 19K-token example is public-domain text (U.S. founding documents, Project Gutenberg eBooks 1, 5, 2 and 1404); the
receipt is made up, drawn for the model card. Code: Apache-2.0. The model is Apache-2.0. It was trained in part on
soft labels distilled from TypeSafe Jev 1.14 via its public API; some of its training data has restrictive or unclear
terms and some training rows are outputs of OpenAI GPT and Anthropic Claude models; see
[Training data and licences](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4#training-data-and-licences)
on the model card. Not affiliated with TypeSafe or Jev. AI disclosure: code written with AI coding assistants
(Claude Code) under my direction; I designed the project, trained the models and verified the results.
