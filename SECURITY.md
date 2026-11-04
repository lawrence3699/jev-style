# Security policy

Jev-Style ships a local HTTP server, a Claude Code `PreToolUse` guard and an MCP server, so a security problem in
it can matter even though nothing leaves your machine by default.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately through GitHub's
[private vulnerability report form](https://github.com/lawrence3699/jev-style/security/advisories/new) for this
repository. Include the package version (`jev-style --version`), the command or request that shows the problem,
and what an attacker could do with it. You will get an answer from the maintainer, Chaoliang Yan, within 7 days;
fixes ship as a new PyPI release and are noted in `CHANGELOG.md`.

## Scope

In scope: the `jev-style` package on PyPI and this repository (server, client, CLI, guard, MCP server, agent skills
and the Claude Code plugins), and the pinned model runtimes it downloads from `chaoliangUNSW/*` on the Hugging Face
Hub. The hosted API at `api.jevstyle.com` and the website are separate services; report problems with them the
same way and they will be routed.

## What the defaults assume

- `jev-style serve` binds to `127.0.0.1`, answers only requests addressed to loopback names (or its `--host`), and
  refuses browser requests from other origins. Exposing it with `--host 0.0.0.0` should go together with
  `--api-key-env`.
- The guard sends the full tool call to the configured server, which must be loopback unless
  `allow_remote_server` is set; its verdict on any failure is "ask", never "allow".
- `JEV_STYLE_API_KEY` is sent only to loopback servers and to the server in `JEV_STYLE_URL`.
- Model files are downloaded at pinned revisions; `jev-style download` checks them against the repo's manifest.

Supported versions: the latest release on PyPI.
