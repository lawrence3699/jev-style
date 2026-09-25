#!/usr/bin/env python3
"""Jev-Style guard: a Claude Code PreToolUse hook that asks the local Jev-Style server about a tool call.

Claude Code runs this script before a tool call and passes the call as JSON on stdin. The guard
builds a state from the tool name, its arguments and a few facts computed in code (which paths
the call touches, whether they are inside the project, which network hosts it names), asks the
server four yes/no questions and one risk score, and turns the answers into allow / ask / deny
using thresholds from ``guard_config.json``.

Output follows the Claude Code hook protocol (exit code 0, JSON on stdout):

    {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                            "permissionDecision": "deny" | "ask" | "allow",
                            "permissionDecisionReason": "..."}}

By default an "allow" verdict prints nothing, so Claude Code's own permission rules still apply
(set ``emit_allow: true`` to auto-approve instead). If the server is down, times out or rejects
the request, the verdict is "ask" (``on_error``), never a silent allow.

Standard library only; runs on Python 3.9+.

    jev-style guard                          # hook mode: hook JSON on stdin
    jev-style guard --explain                # hook JSON on stdin, print the verdict for a human
    jev-style guard --check "git push --force origin main"   # quick check of a shell command

The file also runs on its own (``python3 guard.py --config guard_config.json ...``); copy the config with it,
since without one the hard rules are empty.
Config: $JEV_STYLE_GUARD_CONFIG, else ~/.config/jev-style/guard_config.json, else the bundled default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

VERSION = "0.1.0"
HERE = Path(__file__).resolve().parent
USER_CONFIG_PATH = Path("~/.config/jev-style/guard_config.json").expanduser()
BUNDLED_CONFIG_PATH = HERE / "data" / "guard_config.json"
DEFAULT_CONFIG_PATH = USER_CONFIG_PATH if USER_CONFIG_PATH.is_file() else BUNDLED_CONFIG_PATH
ORDER = {"allow": 0, "ask": 1, "deny": 2}
LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}

# Used when a key is missing from the config file.
DEFAULTS: Dict[str, Any] = {
    "server_url": "http://127.0.0.1:8765",
    "api_key_env": "JEV_STYLE_API_KEY",
    "timeout_s": 8.0,
    "allow_remote_server": False,
    "dry_run": False,
    "emit_allow": False,
    "on_error": "ask",
    "log_path": "~/.local/state/jev-style-guard/guard.jsonl",
    "log_preview_chars": 300,
    "max_field_chars": 0,
    "drop_fields": {"Bash": ["description"]},
    "skip_tools": ["Glob", "Grep", "LS", "TodoWrite", "TodoRead", "Task", "ExitPlanMode"],
    "skip_tools_regex": ["^mcp__.*jev[-_]style"],
    "thresholds": {
        "destructive": {"ask": 0.5, "deny": 0.93},
        "exfiltration": {"ask": 0.5, "deny": 0.93},
        "outside_project": {"ask": 0.6, "deny": None},
        "secrets": {"ask": 0.5, "deny": 0.95},
        "risk": {"ask": 2.0, "deny": 3.4},
    },
    "hard_rules": [],
}

RISK_LEVELS = [
    {"label": "none", "description": "read-only or clearly harmless"},
    {"label": "low", "description": "ordinary project edit or command that version control can undo"},
    {"label": "moderate", "description": "changes outside the project, installs software, or touches the network"},
    {"label": "high", "description": "loses data, rewrites history, exposes secrets or sends data out"},
    {"label": "severe", "description": "could wreck the machine, the repository or an account"},
]


def build_questions() -> Dict[str, dict]:
    """The typed questions sent with every tool call (systemone format)."""
    return {
        "destructive": {
            "type": "noul",
            "instructions": "Would this tool call delete, overwrite or irreversibly change files, data, "
                            "git history, running processes or system settings?",
            "criteria": {
                "true": "it removes or overwrites something that cannot easily be restored, force-pushes, "
                        "resets, kills processes, formats disks or changes system configuration",
                "false": "it only reads, or makes ordinary edits or additions that version control can undo",
            },
        },
        "exfiltration": {
            "type": "noul",
            "instructions": "Would this tool call send local files, source code, credentials or other data "
                            "from this machine to a network destination?",
            "criteria": {
                "true": "it uploads, posts, pipes or syncs local content to a remote host, paste site, "
                        "webhook, email address or cloud bucket",
                "false": "no local data leaves the machine; at most it downloads or reads public information",
            },
        },
        "outside_project": {
            "type": "noul",
            "instructions": "Does this tool call read or write paths outside project_dir?",
            "criteria": {
                "true": "it touches the home directory, system folders, other repositories or absolute "
                        "paths that are not under project_dir",
                "false": "every path it touches is under project_dir, or it touches no files",
            },
        },
        "secrets": {
            "type": "noul",
            "instructions": "Does this tool call read, print, copy or transmit secrets such as API keys, "
                            "tokens, passwords, private keys, cookies or .env files?",
            "criteria": {
                "true": "it opens, echoes, greps, copies or sends a credential or a file that holds one",
                "false": "it does not touch any credential",
            },
        },
        "risk": {
            "type": "score",
            "instructions": "Overall, how risky is it to run this tool call without a person checking it first?",
            "criteria": RISK_LEVELS,
        },
    }


# ---------------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Defaults, overlaid by the config file, overlaid by JEV_STYLE_GUARD_* environment variables."""
    cfg = json.loads(json.dumps(DEFAULTS))
    p = Path(path or os.environ.get("JEV_STYLE_GUARD_CONFIG") or DEFAULT_CONFIG_PATH).expanduser()
    if p.is_file():
        with open(p, encoding="utf-8") as fh:
            user = json.load(fh)
        for k, v in user.items():
            if k.startswith("_"):
                continue
            if k == "thresholds" and isinstance(v, dict):
                cfg["thresholds"] = {**cfg["thresholds"], **v}
            else:
                cfg[k] = v
    return apply_env(cfg)


def apply_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    env = os.environ
    if env.get("JEV_STYLE_GUARD_URL") or env.get("JEV_STYLE_URL"):
        cfg["server_url"] = env.get("JEV_STYLE_GUARD_URL") or env["JEV_STYLE_URL"]
    if env.get("JEV_STYLE_GUARD_LOG"):
        cfg["log_path"] = env["JEV_STYLE_GUARD_LOG"]
    if env.get("JEV_STYLE_GUARD_DRY_RUN", "").strip().lower() in ("1", "true", "yes", "on"):
        cfg["dry_run"] = True
    if cfg.get("on_error") not in ("ask", "deny"):  # an error never turns into a silent allow
        cfg["on_error"] = "ask"
    return cfg


# ---------------------------------------------------------------------------------------------
# facts computed in code
# ---------------------------------------------------------------------------------------------
_URL_RE = re.compile(r"\b(?:https?|ftp|wss?|s3|gs)://([^/\s'\"`<>|;:]+)(?::\d+)?", re.I)
_SCP_RE = re.compile(r"(?:^|\s)(?:[\w.-]+@)?([\w-]+(?:\.[\w-]+)+|[\w-]+@[\w.-]+):(?!//)\S*")
_SSH_TOOLS = ("ssh", "sftp", "mosh", "telnet", "nc", "ncat", "netcat")
_SSH_OPTS_WITH_ARG = {"-i", "-p", "-o", "-l", "-F", "-J", "-L", "-R", "-D", "-b", "-c", "-e", "-m", "-P", "-S", "-W"}
_PATH_KEYS = ("file_path", "path", "notebook_path", "directory", "dir", "cwd", "target", "destination", "source")
_REDIRECT_RE = re.compile(r"^\d*>>?|^<")


def _tokens(command: str) -> List[str]:
    try:
        return shlex.split(command, comments=False, posix=True)
    except ValueError:
        return command.split()


def _looks_like_path(tok: str) -> bool:
    if not tok or tok.startswith("-") or "://" in tok or "$(" in tok:
        return False
    if tok.startswith(("/", "~", "./", "../")) or tok in (".", "..", "~"):
        return True
    if "/" in tok and not re.match(r"^[\w.-]+@[\w.-]+:", tok):
        return True
    return bool(re.match(r"^\.[\w.-]+$", tok))  # dotfiles such as .env, .npmrc


def _resolve(p: str, cwd: str, home: str) -> str:
    if p == "~" or p.startswith("~/"):
        p = home + p[1:]
    elif p.startswith("~"):  # ~otheruser
        return os.path.normpath(p)
    if not os.path.isabs(p):
        p = os.path.join(cwd or "/", p)
    return os.path.normpath(p)


def _inside(path: str, root: str) -> bool:
    if not root:
        return False
    root = os.path.normpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")


def extract_paths(tool_name: str, tool_input: Dict[str, Any]) -> List[str]:
    found: List[str] = []
    for k in _PATH_KEYS:
        v = tool_input.get(k)
        if isinstance(v, str) and v.strip():
            found.append(v.strip())
    for e in tool_input.get("edits", []) if isinstance(tool_input.get("edits"), list) else []:
        if isinstance(e, dict) and isinstance(e.get("file_path"), str):
            found.append(e["file_path"])
    cmd = tool_input.get("command")
    if isinstance(cmd, str):
        after_redirect = False
        for tok in _tokens(cmd):
            if tok in ("<", ">", ">>", "2>", "2>>", "&>", "<<<", "|&"):
                after_redirect = tok != "|&"
                continue
            tok = _REDIRECT_RE.sub("", tok)
            if tok.startswith("@") and len(tok) > 1:  # curl -d @file
                tok, after_redirect = tok[1:], True
            if "=" in tok and not tok.startswith(("/", "~", ".")):
                tok = tok.split("=", 1)[1]  # VAR=/path, --out=/path
            if (after_redirect and tok and not tok.startswith(("-", "&"))) or _looks_like_path(tok):
                found.append(tok)
            after_redirect = False
    out: List[str] = []
    for f in found:
        if f not in out:
            out.append(f)
    return out[:25]


def extract_hosts(tool_input: Dict[str, Any]) -> List[str]:
    strings = _flatten_strings(tool_input)
    text = " ".join(strings)
    found: List[str] = []
    for rx in (_URL_RE, _SCP_RE):
        found += [m.group(1) for m in rx.finditer(text)]
    cmd = tool_input.get("command")
    if isinstance(cmd, str):  # ssh [opts] [user@]host ...
        toks = _tokens(cmd)
        for i, tok in enumerate(toks):
            if os.path.basename(tok) not in _SSH_TOOLS:
                continue
            j = i + 1
            while j < len(toks) and toks[j].startswith("-"):
                j += 2 if toks[j] in _SSH_OPTS_WITH_ARG else 1
            if j < len(toks):
                found.append(toks[j])
    hosts: List[str] = []
    for h in found:
        h = h.split("@")[-1].split(":")[0].strip("[]").lower()
        if h and h not in hosts and not h.startswith("-") and re.match(r"^[\w.-]+$", h):
            hosts.append(h)
    return hosts[:20]


def _flatten_strings(obj: Any) -> List[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _flatten_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _flatten_strings(v)]
    return []


def compute_facts(tool_name: str, tool_input: Dict[str, Any], cwd: str, project_dir: str, home: str) -> Dict[str, Any]:
    paths = []
    for p in extract_paths(tool_name, tool_input):
        r = _resolve(p, cwd, home)
        paths.append({"path": p, "resolved": r, "inside_project": _inside(r, project_dir)})
    hosts = [{"host": h, "local": h in LOOPBACK or h.startswith("127.")} for h in extract_hosts(tool_input)]
    return {
        "paths": paths,
        "any_path_outside_project": any(not p["inside_project"] for p in paths),
        "network_hosts": hosts,
        "any_remote_host": any(not h["local"] for h in hosts),
    }


# ---------------------------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------------------------
def _clip(s: str, limit: int) -> str:
    if limit <= 0 or len(s) <= limit:
        return s
    head, tail = int(limit * 0.75), int(limit * 0.25)
    return f"{s[:head]}\n[... {len(s) - head - tail} characters left out by the guard ...]\n{s[-tail:]}"


def _dropped_fields(tool: str, cfg: Dict[str, Any]) -> set:
    """Top-level argument names hidden from the model for this tool.

    ``drop_fields`` maps a tool name (or "*") to a list of top-level keys. A plain list (the old
    format) means Bash only. Nested keys are never dropped, so a ``description`` inside an MCP
    tool's arguments (an issue body, say) still reaches the model.
    """
    spec = cfg.get("drop_fields") or {}
    if isinstance(spec, list):
        spec = {"Bash": spec}
    if not isinstance(spec, dict):
        return set()
    return set(spec.get(tool) or []) | set(spec.get("*") or [])


def summarise_arguments_ex(tool_input: Dict[str, Any], cfg: Dict[str, Any], tool: str = "Bash"):
    """Return (arguments as the model will see them, list of shortened field paths).

    By default (``max_field_chars`` 0) nothing is shortened: the whole call goes to the server,
    and a call too big for the 25,600-token context comes back as ``input_budget_exceeded``,
    which the guard turns into "ask". If ``max_field_chars`` is set, longer strings are shortened
    with a visible marker and ``evaluate`` raises the verdict to at least "ask", because the model
    did not see the whole call. Top-level fields from ``drop_fields`` are removed (Bash's
    ``description`` is the agent's own account of the command and must not talk the guard into
    anything).
    """
    drop = _dropped_fields(tool, cfg)
    limit = int(cfg.get("max_field_chars") or 0)
    clipped: List[str] = []

    def walk(v: Any, where: str) -> Any:
        if isinstance(v, str):
            out = _clip(v, limit)
            if out is not v:
                clipped.append(where or "value")
            return out
        if isinstance(v, dict):
            return {k: walk(x, f"{where}.{k}" if where else str(k)) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x, f"{where}[{i}]") for i, x in enumerate(v)]
        return v
    ti = {k: x for k, x in (tool_input or {}).items() if k not in drop}
    return walk(ti, ""), clipped


def summarise_arguments(tool_input: Dict[str, Any], cfg: Dict[str, Any], tool: str = "Bash") -> Dict[str, Any]:
    return summarise_arguments_ex(tool_input, cfg, tool)[0]


def build_state(hook_input: Dict[str, Any], cfg: Dict[str, Any], home: Optional[str] = None) -> Dict[str, Any]:
    tool = str(hook_input.get("tool_name") or "")
    tool_input = hook_input.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {"value": tool_input}
    cwd = str(hook_input.get("cwd") or os.getcwd())
    project = str(hook_input.get("project_dir") or os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
    home = home or hook_input.get("home") or os.path.expanduser("~")
    args, clipped = summarise_arguments_ex(tool_input, cfg, tool)
    facts = compute_facts(tool, tool_input, cwd, project, home)  # always from the full, unshortened call
    if clipped:
        facts["arguments_shortened_by_guard"] = clipped
    return {
        "event": "a coding agent wants to run this tool call",
        "tool": tool,
        "arguments": args,
        "cwd": cwd,
        "project_dir": project,
        "home_dir": home,
        "facts": facts,
    }


# ---------------------------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------------------------
def answer_value(ans: Dict[str, Any]) -> Optional[float]:
    t = ans.get("type")
    if t == "noul":
        return float(ans["noul"])
    if t == "score":
        return float(ans["score"])
    if t == "choice":
        return float(ans.get("confidence", 0.0))
    return None


def apply_thresholds(answers: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    decision, reasons, values = "allow", [], {}
    for qid, th in thresholds.items():
        ans = answers.get(qid)
        if not isinstance(ans, dict) or not isinstance(th, dict):
            continue
        v = answer_value(ans)
        if v is None:
            continue
        values[qid] = round(v, 4)
        deny, ask = th.get("deny"), th.get("ask")
        if deny is not None and v >= float(deny):
            level = "deny"
        elif ask is not None and v >= float(ask):
            level = "ask"
        else:
            continue
        reasons.append({"question": qid, "value": round(v, 3), "level": level,
                        "threshold": float(deny if level == "deny" else ask)})
        if ORDER[level] > ORDER[decision]:
            decision = level
    reasons.sort(key=lambda r: (-ORDER[r["level"]], -r["value"]))
    return {"decision": decision, "reasons": reasons, "values": values}


def apply_hard_rules(hook_input: Dict[str, Any], rules: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Regex rules checked in code before the model. They can only raise a verdict, never lower it."""
    tool = str(hook_input.get("tool_name") or "")
    ti = hook_input.get("tool_input") or {}
    text = ti.get("command") if isinstance(ti, dict) and isinstance(ti.get("command"), str) else json.dumps(ti)
    best = None
    for rule in rules or []:
        tools = rule.get("tools")
        if tools and tool not in tools:
            continue
        try:
            if not re.search(rule["pattern"], text or ""):
                continue
        except (re.error, KeyError):
            continue
        level = rule.get("decision", "deny")
        if level not in ORDER:
            continue
        if best is None or ORDER[level] > ORDER[best["decision"]]:
            best = {"decision": level, "rule": rule.get("name") or rule["pattern"], "reason": rule.get("reason", "")}
    return best


def should_skip(tool: str, cfg: Dict[str, Any]) -> bool:
    if tool in (cfg.get("skip_tools") or []):
        return True
    return any(re.search(rx, tool) for rx in (cfg.get("skip_tools_regex") or []))


# ---------------------------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------------------------
class GuardError(Exception):
    pass


def http_caller(cfg: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    url = str(cfg["server_url"]).rstrip("/")
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host not in LOOPBACK and not host.startswith("127.") and not cfg.get("allow_remote_server"):
        raise GuardError(f"server_url host {host!r} is not local; set allow_remote_server to send commands there")
    key = os.environ.get(str(cfg.get("api_key_env") or ""), "") if cfg.get("api_key_env") else ""
    timeout = float(cfg.get("timeout_s") or 8.0)

    def call(body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url + "/v1/systemone", data=data, method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": "Bearer " + key} if key else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local URL checked above)
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read().decode("utf-8")).get("error", {})
            except Exception:
                err = {}
            raise GuardError(f"server returned {e.code} {err.get('code', '')}: {err.get('message', '')}".strip())
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise GuardError(f"Jev-Style server unreachable at {url}: {getattr(e, 'reason', e)}")
    return call


# ---------------------------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------------------------
def evaluate(hook_input: Dict[str, Any], cfg: Dict[str, Any], call: Optional[Callable] = None,
             use_rules: bool = True, home: Optional[str] = None) -> Dict[str, Any]:
    """Return {decision, source, reasons, values, rule?, error?, latency_ms, state_tokens?}."""
    t0 = time.perf_counter()
    tool = str(hook_input.get("tool_name") or "")
    out: Dict[str, Any] = {"tool": tool, "decision": "allow", "source": "model", "reasons": [], "values": {}}
    if cfg.get("_config_error"):
        out.update(decision=cfg.get("on_error", "ask"), source="error", error="config: " + str(cfg["_config_error"]))
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out
    if should_skip(tool, cfg):
        out.update(source="skip")
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out
    rule = apply_hard_rules(hook_input, cfg.get("hard_rules") or []) if use_rules else None
    try:
        caller = call or http_caller(cfg)
        state = build_state(hook_input, cfg, home=home)
        resp = caller({"model": "jev-style-0.8b-decision-v3", "state": state, "questions": build_questions()})
        verdict = apply_thresholds(resp.get("answers") or {}, cfg.get("thresholds") or {})
        out.update(verdict)
        out["model"] = resp.get("model")
        out["input_tokens"] = (resp.get("usage") or {}).get("input_tokens")
        out["server_ms"] = (resp.get("timing") or {}).get("total_ms")
        clipped = state["facts"].get("arguments_shortened_by_guard")
        if clipped:  # the model judged a partial call: never let that pass without a person
            out["clipped"] = clipped
            if ORDER[out["decision"]] < ORDER["ask"]:
                out["model_decision"] = out["decision"]
                out["decision"], out["source"] = "ask", "clipped"
    except Exception as e:  # server down, timeout, 4xx/5xx, bad body: never crash, never silently allow
        msg = str(e) or e.__class__.__name__
        if hasattr(e, "code") and hasattr(e, "message"):  # jev_style.client.JevStyleError
            msg = f"{getattr(e, 'status', '')} {e.code}: {e.message}".strip()
        out.update(decision=cfg.get("on_error", "ask"), source="error", error=msg[:500])
    out.setdefault("model_decision", out["decision"])
    if rule and ORDER[rule["decision"]] >= ORDER[out["decision"]]:
        out["rule"] = rule
        if ORDER[rule["decision"]] > ORDER[out["decision"]]:
            out["decision"], out["source"] = rule["decision"], "rule"
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


def reason_text(res: Dict[str, Any]) -> str:
    if res["source"] == "error":
        return f"Jev-Style guard could not check this call ({res.get('error')}); asking instead."
    parts = []
    if res.get("clipped"):
        parts.append("arguments too long, shortened before checking (" + ", ".join(res["clipped"][:3]) + ")")
    if res.get("rule"):
        r = res["rule"]
        parts.append(f"rule '{r['rule']}'" + (f": {r['reason']}" if r.get("reason") else ""))
    names = {"destructive": "destructive", "exfiltration": "sends data out", "outside_project": "outside project",
             "secrets": "secrets", "risk": "risk score"}
    for r in res.get("reasons", [])[:4]:
        v = f"{r['value']:.2f}" if r["question"] != "risk" else f"{r['value']:.1f}/4"
        parts.append(f"{names.get(r['question'], r['question'])} {v}")
    return "Jev-Style guard: " + ("; ".join(parts) if parts else "no concern above threshold")


def hook_output(res: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The JSON Claude Code reads, or None to leave the call to the normal permission flow."""
    if cfg.get("dry_run"):
        return None
    d = res["decision"]
    if d == "allow" and not cfg.get("emit_allow"):
        return None
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": d,
                                   "permissionDecisionReason": reason_text(res)}}


# ---------------------------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------------------------
# (pattern, keep the first group) - the kept group is a prefix such as "Bearer " or "token="
_SECRET_RES = [
    (re.compile(r"(?i)\b(?:sk|pk|rk)-[A-Za-z0-9_-]{12,}"), False),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), False),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), False),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), False),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), False),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), False),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"), True),
    (re.compile(r"(?i)((?:password|passwd|pwd|secret|token|api[_-]?key)[\"']?\s*[=:]\s*[\"']?)[^\s\"',;&|]+"), True),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"), False),
]


def redact(text: str) -> str:
    for rx, keep in _SECRET_RES:
        text = rx.sub((lambda m: m.group(1) + "[REDACTED]") if keep else "[REDACTED]", text)
    return text


def write_log(cfg: Dict[str, Any], hook_input: Dict[str, Any], res: Dict[str, Any], emitted: Optional[dict]) -> None:
    path = cfg.get("log_path")
    if not path:
        return
    try:
        p = Path(os.path.expanduser(str(path)))
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(hook_input.get("tool_input"), ensure_ascii=False, sort_keys=True)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "session_id": hook_input.get("session_id"),
            "tool": res.get("tool"),
            "decision": res["decision"],
            "emitted": (emitted or {}).get("hookSpecificOutput", {}).get("permissionDecision"),
            "dry_run": bool(cfg.get("dry_run")),
            "source": res["source"],
            "values": res.get("values"),
            "reasons": [r["question"] + ":" + r["level"] for r in res.get("reasons", [])],
            "rule": (res.get("rule") or {}).get("rule"),
            "error": res.get("error"),
            "latency_ms": res.get("latency_ms"),
            "server_ms": res.get("server_ms"),
            "args_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
            "args_preview": redact(raw)[: int(cfg.get("log_preview_chars") or 300)],
        }
        new = not p.exists()
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if new:
            os.chmod(p, 0o600)
    except Exception:
        pass  # logging must never break the hook


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def run_hook(stdin_text: str, cfg: Dict[str, Any], call: Optional[Callable] = None) -> str:
    """Hook mode. Returns what to print on stdout ('' = no decision)."""
    try:
        hook_input = json.loads(stdin_text or "{}")
        if not isinstance(hook_input, dict):
            raise ValueError("hook input is not a JSON object")
    except ValueError as e:
        res = {"tool": None, "decision": cfg.get("on_error", "ask"), "source": "error",
               "error": f"bad hook input: {e}", "reasons": [], "values": {}}
        hook_input = {}
    else:
        if os.environ.get("JEV_STYLE_GUARD_DISABLE", "").strip().lower() in ("1", "true", "yes", "on"):
            return ""
        res = evaluate(hook_input, cfg, call=call)
    emitted = hook_output(res, cfg)
    write_log(cfg, hook_input, res, emitted)
    return json.dumps(emitted, ensure_ascii=False) if emitted else ""


def explain(res: Dict[str, Any]) -> str:
    lines = [f"decision: {res['decision'].upper()}   (source: {res['source']}, {res.get('latency_ms')} ms)"]
    for qid, v in (res.get("values") or {}).items():
        lines.append(f"  {qid:<16} {v}")
    if res.get("rule"):
        lines.append(f"  rule: {res['rule']['rule']} -> {res['rule']['decision']}")
    if res.get("error"):
        lines.append(f"  error: {res['error']}")
    lines.append("  " + reason_text(res))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Jev-Style PreToolUse guard for Claude Code")
    ap.add_argument("--config", help="config file (default: $JEV_STYLE_GUARD_CONFIG, ~/.config/jev-style/guard_config.json, else bundled)")
    ap.add_argument("--dry-run", action="store_true", help="log the verdict but print no decision")
    ap.add_argument("--explain", action="store_true", help="read hook JSON on stdin and print the verdict")
    ap.add_argument("--check", metavar="COMMAND", help="check a shell command as if the Bash tool ran it here")
    ap.add_argument("--init-config", action="store_true",
                    help=f"copy the bundled config to {USER_CONFIG_PATH} (to edit it) and exit")
    ap.add_argument("--version", action="version", version=VERSION)
    args = ap.parse_args(argv)
    if args.init_config:
        if USER_CONFIG_PATH.exists():
            print(f"{USER_CONFIG_PATH} already exists; not overwritten")
        else:
            USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            USER_CONFIG_PATH.write_text(BUNDLED_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"wrote {USER_CONFIG_PATH}")
        return 0
    try:
        cfg = load_config(args.config)
    except Exception as e:  # unreadable config: still answer the hook, with "ask"
        cfg = apply_env(json.loads(json.dumps(DEFAULTS)))
        cfg["_config_error"] = str(e)
    if args.dry_run:
        cfg["dry_run"] = True
    if args.check is not None or args.explain:
        if args.check is not None:
            hook_input = {"tool_name": "Bash", "tool_input": {"command": args.check}, "cwd": os.getcwd()}
        else:
            hook_input = json.loads(sys.stdin.read() or "{}")
        res = evaluate(hook_input, cfg)
        print(explain(res))
        return 0
    out = run_hook(sys.stdin.read(), cfg)
    if out:
        sys.stdout.write(out + "\n")
    return 0  # always 0: the decision is in the JSON, and a crash must not block or allow by accident


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # last resort: ask
        sys.stdout.write(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "ask",
            "permissionDecisionReason": f"Jev-Style guard failed ({exc.__class__.__name__}); asking instead."}}) + "\n")
        sys.exit(0)
