---
name: jev-style-guard
description: Install the Jev-Style guard, a Claude Code PreToolUse hook that has a local decision model check every Bash, Write, Edit, Read, WebFetch and MCP tool call for destructive actions, data exfiltration, secrets access and out-of-project paths, and answers allow / ask / deny. Use when the user wants a local safety net or approval gate for Claude Code (or another agent with PreToolUse-style hooks), wants to block rm -rf / force-push / key exfiltration, or wants to tune or measure such a guard.
---

# Jev-Style guard for Claude Code

Before Claude Code runs a tool, the hook `jev-style guard` sends the call (tool name, full arguments,
cwd, project dir, plus paths and hosts extracted in code) to the local Jev-Style server and asks four
yes/no questions (`destructive`, `exfiltration`, `outside_project`, `secrets`) and one 0-4 `risk` score.
Thresholds turn answers into allow / ask / deny; regex `hard_rules` in code can only make a verdict
stricter. The strictest verdict wins.

Safety properties to tell the user: failures (server down, timeout, bad input, broken config) give
**ask**, never a silent allow; the hook always exits 0; by default an "allow" prints nothing, so Claude
Code's own permission rules still apply; nothing leaves the machine (the server must be loopback).

Measured on the 49 bundled hand-labelled tool calls (Jev-Style-0.8B-Decision-v3, default config):
77.6 % agreement with the labels; no call labelled deny was allowed; 2 of 16 calls labelled ask were
allowed. It is a second line of defence, not a sandbox. Projects under `/tmp` or other system folders get
more "ask" verdicts (the model reads them as outside a normal project); under the home folder,
routine commands (`ls`, `git status`, `npm test`, `pytest`, reading files) are allowed.

## 1. Prerequisites

```sh
command -v jev-style || echo "install first: jev-style-serve skill"
curl -s http://127.0.0.1:8765/healthz      # the guard needs the server running
jev-style guard --check "git push --force origin main"      # expect DENY (rule force-push-main)
jev-style guard --check "ls -la"                             # expect ALLOW
```

The guard needs the server running whenever Claude Code runs; offer to start it at login (see the
`jev-style-serve` skill, step 6). Otherwise every call gets "ask".

## 2. Choose where to install

Ask the user which scope they want:

- **One project**: `.claude/settings.json` in the project (shared with the team) or
  `.claude/settings.local.json` (just them).
- **Every project**: `~/.claude/settings.json`.
- **As a plugin** (easy to toggle): in Claude Code run
  `/plugin marketplace add lawrence3699/jev-style` then `/plugin install jev-style-guard@jev-style`.
  Skip step 3 in that case.

## 3. Add the hook

Merge this into the chosen settings file. Read the file first and keep every existing key and hook;
if `hooks.PreToolUse` already exists, append to the list. Use the absolute path of `jev-style`
(`command -v jev-style`) because hooks do not always inherit the shell PATH.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit|Read|WebFetch|mcp__.*",
        "hooks": [
          {
            "type": "command",
            "command": "/ABSOLUTE/PATH/TO/jev-style guard || printf '%s\\n' '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"ask\",\"permissionDecisionReason\":\"Jev-Style guard could not start; asking instead.\"}}'",
            "timeout": 20
          }
        ]
      }
    ]
  }
}
```

If writing the file fails with "Operation not permitted" (a locked settings file), do not try to
unlock it; tell the user and offer the project-level file or the plugin instead.

## 4. Recommend a dry run first

For the first day, log verdicts without acting on them:

```sh
jev-style guard --init-config        # writes ~/.config/jev-style/guard_config.json (never overwrites)
```

Then set `"dry_run": true` in `~/.config/jev-style/guard_config.json`, or export `JEV_STYLE_GUARD_DRY_RUN=1`
in the environment Claude Code starts from. Verdicts go to `~/.local/state/jev-style-guard/guard.jsonl`
(mode 600; arguments stored as a hash plus a redacted preview). Review it with the user, then set
`dry_run` back to `false`.

## 5. Verify inside Claude Code

Restart Claude Code (hooks load at start). Ask it to run `ls` (should run normally) and
`git push --force origin main` in a scratch repo (should be denied with a "Jev-Style guard: rule
'force-push-main'" reason). `/hooks` in an interactive terminal session lists the installed hook.

## 6. Tune and measure

- Thresholds and rules live in the config (`thresholds.<question>.ask/deny`, `hard_rules`, `skip_tools`).
  Lower an `ask` threshold to be more cautious; raise it if the user gets too many prompts.
- Measure any change on labelled calls: `jev-style guard-replay --url http://127.0.0.1:8765 -v`
  (bundled 49 examples) or `--examples my_calls.jsonl` with rows
  `{"id", "label": "allow|ask|deny", "hook_input": {"tool_name", "tool_input", "cwd", "project_dir", "home"}}`
  (set `project_dir` and `home` explicitly; otherwise they default to `$CLAUDE_PROJECT_DIR` / the cwd).
- Turn it off temporarily: `JEV_STYLE_GUARD_DISABLE=1`.
