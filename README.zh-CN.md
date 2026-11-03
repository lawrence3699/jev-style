<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lawrence3699/jev-style/main/docs/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/lawrence3699/jev-style/main/docs/assets/logo.svg" alt="" width="72">
  </picture>
</p>

<h1 align="center">Jev-Style</h1>

<p align="center"><strong>在自己的电脑上跑的小型决策模型。输入文本和图片，输出每个选项的校准概率。</strong></p>

<p align="center">
  <a href="https://github.com/lawrence3699/jev-style/blob/main/README.md">English</a> ·
  <a href="https://jevstyle.com/">官网</a> ·
  <a href="https://huggingface.co/collections/chaoliangUNSW/jev-style-decision-v4-2b">模型</a> ·
  <a href="https://huggingface.co/spaces/chaoliangUNSW/jev-style-2b">在线试用</a> ·
  <a href="https://pypi.org/project/jev-style/">PyPI</a> ·
  <a href="https://github.com/lawrence3699/jev-style/blob/main/CHANGELOG.md">更新日志</a>
</p>

本文是中文摘要，完整说明（API 细节、级联、评测协议、免责声明）以 [英文 README](https://github.com/lawrence3699/jev-style/blob/main/README.md) 为准。作者：[Chaoliang Yan](https://github.com/lawrence3699)。本项目独立维护，与 TypeSafe 和 Jev 无关联。

## 这是什么

Jev-Style 是一组小型决策模型（0.8B 和 2B，基于 Qwen3.5 微调）加上让它们能干活的工具。给模型一段文本或 JSON、可选的图片、以及若干"有类型的问题"，它只读一遍输入，就在一次前向传播里返回每个选项的校准概率，全程在你的笔记本上完成。典型用途：客服工单分流、判断什么时候该调用更大的模型、审核小票和截图、让编程 agent 的工具调用先过一道本地模型的审批。不需要 GPU，不需要 API key，不需要训练。

三种问题类型，一次请求可以混合提问：

| 类型 | 问法 | 返回 |
|---|---|---|
| `noul` | 这句话对这段输入成立吗？ | `noul` = 成立的概率 |
| `choice` | 下面哪个选项？（1 到 255 个） | `choice`、各选项 `probabilities`、`confidence` |
| `score` | 在这个有序量表上是第几档？（2 到 10 档） | `score`（期望档位）、`legend`、`probabilities`、`confidence` |

## 快速开始

先在浏览器里试：[2B Space](https://huggingface.co/spaces/chaoliangUNSW/jev-style-2b)（文本 + 图片）、[0.8B Space](https://huggingface.co/spaces/chaoliangUNSW/jev-style-v3)。

本地需要 Python 3.10 以上，按机器装**一个**后端：

```bash
pip install -U "jev-style[mlx]"      # Apple silicon
pip install -U "jev-style[torch]"    # NVIDIA GPU、CPU，或需要图片输入（Apple MPS 也可以）
```

启动服务，打开本地 Playground（http://127.0.0.1:8765）。第一次启动会下载 0.8B 权重（约 1.5 GB），以后直接复用：

```bash
jev-style serve                      # 加 --release 2b 用 2B v4；图片要加 --backend torch
```

另开一个终端做一次决策：

```bash
jev-style decide "I was charged twice" --url http://127.0.0.1:8765 --choice "Which team?::billing,shipping,returns"
```

Python 一行：

```python
import jev_style

jev_style.classify("Where is my parcel? It was due Monday.", ["billing", "shipping", "tech"])
# {'label': 'shipping', 'confidence': ..., 'probabilities': {...}}
```

多个问题、自己选引擎：

```python
from jev_style import JevStyle, choice, noul, score

js = JevStyle(release="2b")                       # 2B v4，后端按本机自动选；JevStyle() 是默认的 0.8B
out = js.decide("I was charged twice. Please fix this ASAP.", {
    "billing": noul("This ticket is about billing."),
    "tone":    choice("What is the customer's tone?", ["calm", "frustrated", "angry"]),
    "urgency": score("How urgent is this ticket?", ["can wait", "this week", "today"]),
})
print(out["answers"]["billing"]["noul"], out["answers"]["tone"]["choice"])
```

服务端实现公开的 `POST /v1/systemone` 请求格式，为 Jev 兼容服务写的客户端（包括官方 SDK）把地址指向 `http://127.0.0.1:8765` 就能用，API key 随便填。

## 为什么用它

- **概率，不只是标签。** 每个版本都带在留出数据上拟合的温度，代码可以只采纳有把握的答案，其余交给人或大模型。
- **看图（2B v4）。** 照片、截图、小票、图表、扫描表单放进 state，按 TypeSafe API 的 `assets` 方式传。PyTorch 构建能看图，MLX 和 GGUF 构建只读文本。
- **长输入。** 2B v4 每次最多 32,768 token（v3 为 25,600）。超长不会被静默截断，而是报错说明。
- **本地、够小。** 0.8B 的 4-bit 版只有 0.53 GB；M1 Max 上 MLX 短请求约 0.15 到 0.2 秒。数据不出本机。
- **多语言。** 训练覆盖 19 种语言；在 MASSIVE 意图识别上，0.8B 在全部 51 种评测语言上都高于 Laya 的多语言版。
- **为 agent 而建。** 六个 agent skill、官方注册表收录的 MCP server、Claude Code 工具调用守卫、置信度级联，适用于 Claude Code、Codex、Cursor 等。
- **数字可查。** `jev-style eval` 用你自己的标注数据打分；README 里每个基准数字都链接到带协议和置信区间的模型卡。

## 模型

先用默认的 0.8B 上手（1.5 GB，笔记本够用）；要精度、图片和 32k 输入就用 2B v4（`--release 2b`；`--release 2b-v3` 保留上一代 2B）。

| 版本（`--release`） | 参数量 · 最小构建 | JevBench v1.4.1 公开题（231） | tweet_topic 零样本（1,693） | 上下文 |
|---|---|:---:|:---:|:---:|
| [Jev-Style-2B-Decision-v4](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v4)（`2b`） | 2.2B · 1.27 GB（Q4_K_M，仅文本） | **74.9 %** | 未跑 | **32,768 token**，支持图片 |
| [Jev-Style-2B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3)（`2b-v3`） | 1.9B · 1.27 GB（Q4_K_M） | 73.6 % | **82.2 %** | 25,600 token |
| [Jev-Style-0.8B-Decision-v3](https://huggingface.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3)（`0.8b`，默认） | 0.8B · 0.53 GB（Q4_K_M） | 64.1 % | 75.5 % | 25,600 token |
| 托管版 Jev 1.13（参考） | – | 86.6 % | 79.3 % | – |

2B v4 在 Decision Index 0.3 公开套件上自测 44.37（2B v3 榜上为 34.05），公开 Vision 套件 57.55。这些都是用官方工具的单次自测，不是官方榜单条目；部分基准的官方训练集在 v4 的训练数据里，所以那些结果不是零样本。协议、置信区间和训练数据见各模型卡。托管版 Jev 在所有公布过数字的数据集上都更准，这些模型是本地的小型选项，不是替代品。

## Agent 工具

**六个 agent skill**（[`skills/`](https://github.com/lawrence3699/jev-style/tree/main/skills)）：

```bash
npx skills add lawrence3699/jev-style --skill '*' -y
```

| Skill | agent 用它做什么 |
|---|---|
| `jev-style-serve` | 装 CLI、选后端、启动服务、用真实请求验证，可选开机自启 |
| `jev-style` | 写调用模型的代码：请求格式、读概率、按阈值转成动作、限制条件 |
| `jev-style-eval` | 从你的数据构造标注 JSONL 并跑 `jev-style eval`：准确率、Brier、ECE、1/5/10 % 错误预算下可自动化的比例、重拟合温度 |
| `jev-style-adopt` | 扫描代码里只返回标签/是否/评分的 LLM 调用，改写成类型化问题；LLM 保留为低置信度时的兜底；先影子运行、测一致率、再切换 |
| `jev-style-guard` | 安装 Claude Code 守卫 hook，建议先 dry run，再调阈值、用标注调用验证 |
| `jev-style-mcp` | 在 Claude Code、Codex、Cursor、Claude Desktop、Windsurf 注册 MCP 工具并验证 |

Claude Code 里也可以作为插件装：`/plugin marketplace add lawrence3699/jev-style`，然后 `/plugin install jev-style@jev-style`（六个 skill + MCP 工具）和 `/plugin install jev-style-guard@jev-style`（PreToolUse 守卫）。

**Claude Code 守卫。** Claude Code 执行 `Bash`、`Write`、`Edit`、`Read`、`WebFetch` 或 MCP 调用之前，`jev-style guard` 把调用交给本地模型，问它是否有破坏性、是否外泄数据、是否碰到密钥、是否越出项目，外加 0 到 4 的风险分，然后得出 **allow / ask / deny**。代码里的正则硬规则只能让结论更严。服务挂了、超时或拒绝请求时一律 **ask**，绝不静默放行。在 49 条手工标注的工具调用上，默认配置与标注一致 77.6 %，**标注为 deny 的没有一条被放行**。它是第二道防线，不是沙箱。

**MCP server。** `claude mcp add jev-style --scope user -- jev-style mcp`。它是一个很薄的 stdio 服务，转发给正在运行的 `jev-style serve`，一份模型服务所有客户端。已收录在 [官方 MCP Registry](https://registry.modelcontextprotocol.io/)（`io.github.lawrence3699/jev-style`，`uvx jev-style mcp`），提供 `decide`、`noul`、`choice`、`score`、`model_info`。

## 用自己的数据评测

```bash
jev-style eval examples/support_tickets.jsonl --url http://127.0.0.1:8765
```

报告每个问题的准确率、Brier、ECE，以及在 1 %、5 %、10 % 错误预算下能自动处理的比例和对应阈值。`automate @5%: 93.3% (p>=0.57)` 的读法：只在最高概率 ≥ 0.57 时采纳，模型就能处理 93.3 % 的工单且其中错误不超过 5 %，其余交给人或大模型。`--server` 可以重复多次，把同一份文件跑过多个引擎做配对比较。

## 级联、后端与速度

**级联**：每个问题先由小模型回答，置信度低于阈值的才送到更大的模型。内置的 `cascade-9b` = 我们的 2B v3 + 第三方 [JevK5-9B](https://huggingface.co/alibiserikbay/JevK5-9B)，阈值 τ = 0.75 在 4,992 题的校准集上按事先写好的规则冻结；JevBench 公开题 87.9 %（203/231），53 % 的题由 2B 回答，RTX 5090 上每请求中位 29 ms。需要约 32 GB 显存和单独安装的 jevk5 包。

| 机器 | 命令 |
|---|---|
| Apple silicon | `jev-style serve`（MLX bf16；`--precision 8bit` 更省） |
| NVIDIA GPU | `jev-style serve --backend torch`（默认开 CUDA graphs：RTX 5090 上 2B v3 中位延迟 86.2 → 13.9 ms，答案不变） |
| 仅 CPU | `jev-style serve --backend torch --device cpu` |
| Mac 上看图 | `jev-style serve --release 2b --backend torch` |
| llama.cpp | 先用 GGUF 仓库里的 `build_jev_score.sh` 编译打分器，再 `--backend gguf --scorer ...` |
| 离线 | `jev-style serve --model-dir /path/to/model` |
| 无模型（调 UI） | `jev-style serve --fake` |

## 局限

- 托管版 Jev 更准（JevBench 86.6 % 对 2B v4 的 74.9 %）。
- 标注够多时传统分类器可能赢：六意图 BANKING77 对比（240 条测试）中 TF-IDF + 逻辑回归对 229 条，0.8B 对 224 条。没有标注、一次多问、有图片、输入很长、需要可设阈值的概率，才是 Jev-Style 的用武之地。
- 看图只有 2B v4 的 PyTorch 构建支持；原版 llama.cpp、Ollama、LM Studio、`mlx_lm.generate` 能加载权重但算不出决策分数。
- Mac 上短请求是零点几秒，不是几十毫秒；2B 的 MLX 运行时要求 mlx-lm 正好 0.31.3。

## 引用与许可

代码 Apache-2.0；权重是 Qwen3.5-2B / 0.8B 的 Apache-2.0 微调。2B v4 的部分训练标签蒸馏自 TypeSafe Jev 1.14 的公开 API（见其模型卡）；本包不分发 TypeSafe 的权重、代码或原始教师响应数据。与 TypeSafe、Jev、阿里云 Qwen 团队、Laya 作者、JevK5 作者均无关联。引用格式见 [CITATION.cff](https://github.com/lawrence3699/jev-style/blob/main/CITATION.cff)。
