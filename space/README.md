---
title: Jev-Style v3
emoji: ⚖️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
short_description: 0.8B decisions, a probability for every option
models:
  - chaoliangUNSW/Jev-Style-0.8B-Decision-v3
  - chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF
  - chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX
preload_from_hub:
  - chaoliangUNSW/Jev-Style-0.8B-Decision-v3 LICENSE,NOTICE,chat_template.jinja,config.json,generation_config.json,jev_style_decision.py,manifest.json,model.safetensors,readout_config.json,release_config.json,requirements.txt,tokenizer.json,tokenizer_config.json 4635f7eb619ac1683fe9776ec436518f070eb20d
tags:
  - text-classification
  - llm-routing
  - guardrails
  - calibration
---

# Jev-Style v3

Try [Jev-Style-0.8B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3): give it a text
and a question, get a calibrated probability for every option. Choice, yes/no or score; up to 25,600 tokens of input.

Runs the model repo's own PyTorch runtime (float32) on ZeroGPU. Other builds:
[GGUF](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF) (0.53 GB in 4-bit) ·
[MLX](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-MLX). Website: [jevstyle.com](https://jevstyle.com).

**Run it on your own machine:** `pip install "jev-style[torch]"` (`[mlx]` on Apple silicon; [PyPI](https://pypi.org/project/jev-style/)), then `jev-style serve` for a local API,
Playground, agent skills, Claude Code guard and MCP tools: [github.com/lawrence3699/jev-style](https://github.com/lawrence3699/jev-style).

The 19K-token example is public-domain text (U.S. founding documents, Project Gutenberg eBooks 1, 5, 2 and 1404).
Code: Apache-2.0. Not affiliated with TypeSafe, Jev or Laya.
