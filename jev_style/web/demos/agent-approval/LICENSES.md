# Agent action approval: licences and sources

- All code in this folder (`index.html`, `approval.js`, `approval.css`), the server extension
  (`jev_style/ext/agent_approval.py`) and the tests (`its tests`) were written for
  this project. No third-party code, fonts, icons, libraries or CDNs are used.
- The verdicts come from the project's own Claude Code guard hook
  (`jev_style/guard.py` with `guard_config.json`), imported by the extension,
  not copied.
- Tool calls in the session feed: 22 are rows of `integrations/claude-code-guard/examples/labelled_commands.jsonl`
  (hand-written for this project, used verbatim); 2 (`x01`, `x02`) are added in the extension in the same
  style. Host names use reserved example domains (`example.com`, `example.net`, ...) and documentation IP
  ranges (`203.0.113.0/24`, `198.51.100.0/24`). No keys or secrets appear anywhere.
- Nothing from the training pool, `data/eval-v3` or any sealed test suite is used.
- Commands are never executed: the page and the server only handle their text.
