---
name: jev-style-mcp
description: Connect the Jev-Style local decision model to an AI coding agent as an MCP server (tools decide, noul, choice, score, model_info) - Claude Code, Claude Desktop, Cursor, Codex CLI, Windsurf or any MCP client. Use when the user wants their agent to classify, route, gate or score text with a local model through MCP, or asks to add jev-style as an MCP server.
---

# Jev-Style as MCP tools

The MCP server is a thin, stdio process: `jev-style mcp`. By default it forwards to a running
`jev-style serve` on http://127.0.0.1:8765 (so one model in memory serves every client). It refuses
non-loopback URLs unless `--allow-remote` is given.

| Tool | Arguments | Returns |
|---|---|---|
| `decide` | `state`, `questions` (systemone format, many at once) | full response |
| `noul` | `state`, `question`, `true_means?`, `false_means?` | `p_true`, `answer` |
| `choice` | `state`, `question`, `options` (list or `{option: description}`) | `choice`, `probabilities`, `confidence` |
| `score` | `state`, `question`, `levels` (2..10, lowest first) | `score`, `level`, `legend`, `probabilities`, `confidence` |
| `model_info` | none | model id, budgets, backend |

## 1. Prerequisites

```sh
command -v jev-style || echo "install first (jev-style-serve skill)"
jev-style mcp --help >/dev/null && echo "mcp extra ok"
curl -s http://127.0.0.1:8765/healthz || echo "server not running"
```

If `jev-style mcp --help` fails with "needs the 'mcp' extra", reinstall with the extra:
`uv tool install --force "jev-style[all]"`.
If the server is not running, start it (`jev-style serve`, see the `jev-style-serve` skill). Without a
server you can use `jev-style mcp --model` (loads the model inside the MCP process; each client then
holds its own copy) - prefer the shared server.

## 2. Register it

Use the absolute path from `command -v jev-style` if the client does not inherit the shell PATH.

**Claude Code**

```sh
claude mcp add jev-style --scope user -- jev-style mcp
claude mcp list                                   # jev-style ... connected
```

**Codex CLI** (`~/.codex/config.toml`)

```toml
[mcp_servers.jev-style]
command = "jev-style"
args = ["mcp"]
```

**Cursor** (`~/.cursor/mcp.json`), **Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`),
**Windsurf** and other JSON-configured clients:

```json
{"mcpServers": {"jev-style": {"command": "jev-style", "args": ["mcp"]}}}
```

Merge into existing JSON; never overwrite the user's other servers. Server on another port:
`"args": ["mcp", "--url", "http://127.0.0.1:8766"]`. Server with a key: add
`"env": {"JEV_STYLE_API_KEY": "..."}` (ask the user for the value; do not invent one).

## 3. Verify

Restart or reload the client, then call `model_info` - expect `jev-style-0.8b-decision-v3`. Then a real
call, e.g. `choice(state="I was charged twice", question="Which team?", options=["billing","tech","sales"])`
should pick `billing`. If `model_info` says `jev-style-fake`, the server runs the fake engine.

Without a client, the same check from the shell:

```sh
(printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"choice","arguments":{"state":"I was charged twice","question":"Which team?","options":["billing","tech","sales"]}}}'; sleep 5) \
  | jev-style mcp | grep -o '"choice":"[a-z]*"'  # expect "choice":"billing"
```

## Using the tools well

Put everything the judgement needs into `state`. Ask several questions about one state with `decide`.
Treat outputs as probabilities: act on high confidence, ask the user in the middle band.
