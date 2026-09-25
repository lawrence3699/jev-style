/* Agent action approval demo. Plays a coding-agent session (GET /api/agent-approval/feed) one tool call at a
 * time; each call goes to POST /api/agent-approval/check, which runs the Claude Code guard's own evaluate()
 * (jev_style/guard.py): hard rules in code, then Jev-Style v3's answers to four yes/no
 * questions and a risk score, then the guard's thresholds. The page draws what comes back; it never decides
 * anything itself and never runs a command. No build step, no third-party code.
 * URL parameters (optional): speed (0.25..16), autoplay=1.
 * Test hook: window.AgentApproval; <html data-aa-ready> once the feed is loaded. */
(function () {
  "use strict";
  var M, $ = function (id) { return document.getElementById(id); };
  var BARS = ["destructive", "exfiltration", "outside_project", "secrets"];
  var QUESTIONS = BARS.concat(["risk"]);
  var Q_NAMES = { destructive: ["destructive", "破坏性"], exfiltration: ["exfiltration", "外传"],
                  outside_project: ["outside project", "项目外"], secrets: ["secrets", "密钥"], risk: ["risk (0–4)", "风险（0–4）"] };
  var VERDICT = { allow: ["✓", "Allow", "允许"], ask: ["?", "Ask", "询问"], deny: ["✕", "Deny", "拒绝"] };
  var SOURCE = { rule: ["rule", "规则"], model: ["model", "模型"], error: ["error", "错误"], skip: ["skipped", "跳过"],
                 clipped: ["clipped", "截断"] };
  var ABORT = { abort: true };
  var S = {
    feed: null, items: [], ctx: null, idx: 0, playing: false, busy: false, gen: 0, speed: 1, ended: false,
    rows: [], selected: null, pinned: false, counts: { allow: 0, ask: 0, deny: 0 }, ms: [], model: null,
    current: Promise.resolve(true)
  };

  function t(en, zh) { return M.t(en, zh); }
  function bi(en, zh) { return M.bi(en, zh); }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, Math.max(0, ms)); }); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function fmtMs(ms) {
    if (ms == null || !isFinite(ms)) return "–";
    return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : (ms < 10 ? ms.toFixed(1) : String(Math.round(ms))) + " ms";
  }
  function mean(xs) { return xs.length ? xs.reduce(function (a, b) { return a + b; }, 0) / xs.length : null; }
  function json(x) { return JSON.stringify(x, null, 2); }
  function clip(s, n) { s = String(s == null ? "" : s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  // ------------------------------------------------------------------ how a tool call reads in the terminal
  function shortPath(p, item) {
    p = String(p || "");
    var proj = (item.project_dir || "").replace(/\/+$/, ""), home = (item.home || "").replace(/\/+$/, "");
    if (proj && p.indexOf(proj + "/") === 0) return p.slice(proj.length + 1);
    if (home && (p === home || p.indexOf(home + "/") === 0)) return "~" + p.slice(home.length);
    return p;
  }
  function display(item) {
    var i = item.input || {}, tool = item.tool;
    if (tool === "Bash") return { tag: "$", bash: true, text: String(i.command || ""), extra: "" };
    if (tool === "Read") return { tag: "Read", text: shortPath(i.file_path, item), extra: "" };
    if (tool === "Edit" || tool === "MultiEdit") {
      return { tag: "Edit", text: shortPath(i.file_path, item),
               extra: i.old_string != null ? clip(i.old_string, 40) + "  →  " + clip(i.new_string, 40) : "" };
    }
    if (tool === "Write") {
      var n = String(i.content || "").replace(/\n$/, "").split("\n").length;
      return { tag: "Write", text: shortPath(i.file_path, item), extra: "+" + n + t(n === 1 ? " line" : " lines", " 行") };
    }
    if (tool === "WebFetch") return { tag: "Fetch", text: String(i.url || ""), extra: "" };
    if (tool === "NotebookEdit") return { tag: "Notebook", text: shortPath(i.notebook_path, item), extra: clip(i.new_source, 40) };
    return { tag: tool, text: clip(JSON.stringify(i), 200), extra: "" };
  }

  // ------------------------------------------------------------------ rows
  function makeRow(item) {
    var d = display(item), row = el("div", "aa-row");
    row.setAttribute("role", "listitem");
    row.setAttribute("tabindex", "0");
    row.setAttribute("data-state", "typing");
    row.setAttribute("aria-current", "false");
    row.setAttribute("data-tool", item.tool);
    if (item.id) row.setAttribute("data-id", item.id);
    if (item.custom) row.setAttribute("data-custom", "");
    var cmd = el("div", "cmd");
    if (item.custom) cmd.appendChild(el("span", "aa-you")).appendChild(bi("you", "你"));
    cmd.appendChild(el("span", "aa-tool " + (d.bash ? "bash" : "pill"), d.tag));
    var text = el("span", "aa-text");
    cmd.appendChild(text);
    var extra = el("span", "aa-extra");
    cmd.appendChild(extra);
    row.appendChild(cmd);
    var chip = el("div", "chip"), src = el("div", "src"), bars = el("div", "bars aa-bars"), ms = el("div", "ms");
    BARS.forEach(function (q) {
      var b = el("div", "aa-bar");
      b.setAttribute("data-q", q);
      b.appendChild(el("span", "f"));
      bars.appendChild(b);
    });
    row.appendChild(chip); row.appendChild(src); row.appendChild(bars); row.appendChild(ms);
    var entry = { row: row, item: item, d: d, text: text, extra: extra, chip: chip, src: src, bars: bars, ms: ms,
                  request: null, data: null, status: null };
    row.addEventListener("click", function () { select(entry, true); });
    row.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(entry, true); } });
    return entry;
  }
  function nearBottom() { var f = $("aa-feed"); return f.scrollHeight - f.scrollTop - f.clientHeight < 80; }
  function toBottom(force) {
    var f = $("aa-feed");
    if (force || S.follow) f.scrollTop = f.scrollHeight;
  }
  function appendRow(entry) {
    S.follow = nearBottom() || !S.rows.length;
    $("aa-bigplay").hidden = true;
    $("aa-feed").appendChild(entry.row);
    S.rows.push(entry);
    toBottom();
  }
  function typeText(entry, gen) {
    var s = entry.d.text, n = s.length;
    if (!n) return Promise.resolve();
    var per = Math.min(28, Math.max(8, 900 / n)) / S.speed;       // at most ~0.9 s of typing at 1x
    var chunk = Math.max(1, Math.ceil(16 / per)), i = 0;
    return new Promise(function (done, fail) {
      (function next() {
        if (gen !== S.gen) return fail(ABORT);
        i = Math.min(n, i + chunk);
        entry.text.textContent = s.slice(0, i);
        if (i % 20 < chunk) toBottom();
        if (i >= n) return done();
        setTimeout(next, per * chunk);
      })();
    });
  }
  function setWaiting(entry) {
    entry.row.setAttribute("data-state", "checking");
    entry.extra.textContent = entry.d.extra;
    entry.chip.innerHTML = "";
    var w = el("span", "aa-wait");
    w.appendChild(el("i")); w.appendChild(el("i")); w.appendChild(el("i"));
    entry.chip.appendChild(w);
    toBottom();
  }
  function chipEl(decision, big) {
    var v = VERDICT[decision] || ["!", decision, decision];
    var c = el("span", "aa-chip " + (VERDICT[decision] ? decision : "err"));
    c.appendChild(document.createTextNode(v[0] + " "));
    c.appendChild(bi(v[1], v[2]));
    if (big === false) c.className = "mj-chip " + ({ allow: "act", ask: "review", deny: "block" })[decision];
    return c;
  }
  function levelOf(data, q) {
    var r = (data.reasons || []).filter(function (x) { return x.question === q; })[0];
    return r ? r.level : "";
  }
  function barTitle(data, q) {
    var v = (data.values || {})[q], th = (data.thresholds || {})[q] || {};
    if (v == null) return t(Q_NAMES[q][0], Q_NAMES[q][1]) + ": –";
    return t(Q_NAMES[q][0], Q_NAMES[q][1]) + " " + v.toFixed(2) +
      (th.ask != null ? t(" · ask ≥ ", " · 询问 ≥ ") + th.ask : "") + (th.deny != null ? t(" · deny ≥ ", " · 拒绝 ≥ ") + th.deny : "");
  }
  function applyResult(entry, data) {
    entry.data = data;
    var row = entry.row;
    row.setAttribute("data-state", "done");
    row.setAttribute("data-decision", data.decision);
    row.setAttribute("data-source", data.source);
    entry.chip.innerHTML = "";
    entry.chip.appendChild(chipEl(data.decision));
    var s = SOURCE[data.source] || [data.source, data.source];
    entry.src.innerHTML = "";
    var tag = el("span", "aa-src" + (data.source === "rule" ? " rule" : ""));
    tag.appendChild(bi(s[0], s[1]));
    entry.src.appendChild(tag);
    entry.src.title = data.rule ? data.rule.rule + ": " + (data.rule.reason || "") : "";
    BARS.forEach(function (q, k) {
      var b = entry.bars.children[k], v = (data.values || {})[q], th = (data.thresholds || {})[q] || {};
      b.setAttribute("data-level", levelOf(data, q));
      b.title = barTitle(data, q);
      b.querySelectorAll(".tick").forEach(function (x) { x.remove(); });
      ["ask", "deny"].forEach(function (lv) {
        if (th[lv] == null) return;
        var tk = el("span", "tick" + (lv === "deny" ? " deny" : ""));
        tk.style.left = (Math.max(0, Math.min(1, th[lv])) * 100).toFixed(1) + "%";
        b.appendChild(tk);
      });
      var f = b.querySelector(".f");
      requestAnimationFrame(function () { f.style.width = v == null ? "0" : (Math.max(0, Math.min(1, v)) * 100).toFixed(1) + "%"; });
    });
    var total = data.timing && data.timing.total_ms;
    entry.ms.textContent = fmtMs(total);
    entry.chip.title = data.hook_reason || "";
    if (data.decision === "deny") {
      row.classList.remove("flash");
      void row.offsetWidth;
      row.classList.add("flash");
    }
    if (S.counts[data.decision] != null) {
      S.counts[data.decision]++;
      bump("aa-n-" + data.decision, S.counts[data.decision]);
    }
    if (total != null) { S.ms.push(total); if (S.ms.length > 50) S.ms.shift(); }
    if (data.model_name || data.model) S.model = data.model_name || data.model;
    renderStats();
    if (!S.pinned) select(entry, false);
    toBottom();
  }
  function markError(entry, msg) {
    entry.row.setAttribute("data-state", "done");
    entry.row.setAttribute("data-decision", "error");
    entry.chip.innerHTML = "";
    var c = el("span", "aa-chip err", "!");
    entry.chip.appendChild(c);
    entry.chip.title = msg;
  }
  function bump(id, value) {
    var b = $(id);
    b.textContent = value;
    b.classList.remove("bump");
    void b.offsetWidth;
    b.classList.add("bump");
  }

  // ------------------------------------------------------------------ the check (the only place that talks to the server)
  function requestFor(item) {
    if (item.custom) return { tool: item.tool, input: item.input };
    return { tool: item.tool, input: item.input, cwd: item.cwd, project_dir: item.project_dir, home: item.home };
  }
  function postCheck(body) {
    var headers = { "content-type": "application/json" }, key = M.getKey();
    if (key) headers.authorization = "Bearer " + key;
    return fetch("/api/agent-approval/check", { method: "POST", headers: headers, body: JSON.stringify(body) })
      .then(function (r) {
        return r.text().then(function (txt) {
          var data;
          try { data = JSON.parse(txt); } catch (e) { data = { error: { code: "bad_response", message: txt.slice(0, 300) } }; }
          return { ok: r.ok, status: r.status, data: data };
        });
      }, function (e) { return { ok: false, status: 0, data: { error: { code: "network_error", message: String(e) } } }; });
  }
  function errorText(r) {
    var e = (r.data && r.data.error) || {};
    if (r.status === 401) return t("The server needs an API key: set it in the Playground.", "服务需要 API key，请先在试用页填写。");
    if (r.status === 0) return t("Server not reachable: ", "连不上服务：") + (e.message || "");
    return "HTTP " + r.status + ": " + (e.code ? e.code + " – " : "") + (e.message || "");
  }
  function showError(msg) { var e = $("aa-error"); e.hidden = !msg; e.textContent = msg || ""; }

  /* One tool call: type it, check it, show the verdict, hold. Resolves true when it completed. */
  function run(item) {
    var gen = S.gen, entry = makeRow(item);
    S.busy = true;
    renderButtons();
    appendRow(entry);
    var body = requestFor(item);
    entry.request = body;
    var p = (item.custom ? Promise.resolve(entry.text.textContent = entry.d.text) : typeText(entry, gen))
      .then(function () {
        if (gen !== S.gen) throw ABORT;
        setWaiting(entry);
        return Promise.all([postCheck(body), sleep(220 / S.speed)]);
      })
      .then(function (res) {
        if (gen !== S.gen) throw ABORT;
        var r = res[0];
        entry.status = r.status;
        if (!r.ok) {
          markError(entry, errorText(r));
          showError(errorText(r));
          S.playing = false;
          return false;
        }
        showError(null);
        applyResult(entry, r.data);
        var hold = (r.data.decision === "deny" ? 1500 : r.data.decision === "ask" ? 1150 : 950) / S.speed;
        return sleep(hold).then(function () { return true; });
      })
      .catch(function (e) {
        if (e === ABORT) return false;
        if (gen === S.gen) { showError(String(e && e.message || e)); S.playing = false; }
        return false;
      })
      .then(function (ok) {
        if (gen === S.gen) { S.busy = false; renderButtons(); }
        return ok;
      });
    S.current = p;
    return p;
  }

  // ------------------------------------------------------------------ playback
  function loop(gen) {
    if (!S.playing || gen !== S.gen) return;
    if (S.idx >= S.items.length) { finish(); return; }
    run(S.items[S.idx++]).then(function (ok) {
      if (gen !== S.gen) return;
      if (!ok) { S.playing = false; renderButtons(); return; }
      if (S.playing) loop(gen);
      else if (S.idx >= S.items.length) finish();
    });
  }
  function play() {
    if (!S.items.length || S.playing) return;
    if (S.ended) reset();
    S.playing = true;
    renderButtons();
    var gen = S.gen;
    S.current.then(function () { if (gen === S.gen && S.playing && !S.busy) loop(gen); });
  }
  function pause() { S.playing = false; renderButtons(); }
  function step() {
    if (S.playing || S.busy || S.idx >= S.items.length) return Promise.resolve(false);
    var gen = S.gen;
    return run(S.items[S.idx++]).then(function (ok) {
      if (gen === S.gen && S.idx >= S.items.length) finish();
      return ok;
    });
  }
  function reset() {
    S.gen++;
    S.playing = false; S.busy = false; S.ended = false; S.idx = 0; S.pinned = false; S.selected = null;
    S.rows = []; S.counts = { allow: 0, ask: 0, deny: 0 }; S.ms = [];
    S.current = Promise.resolve(true);
    var f = $("aa-feed");
    Array.prototype.slice.call(f.children).forEach(function (c) { if (c.id !== "aa-bigplay") c.remove(); });
    $("aa-bigplay").hidden = false;
    showError(null);
    ["allow", "ask", "deny"].forEach(function (k) { $("aa-n-" + k).textContent = "0"; });
    renderStats();
    renderSent();
    renderButtons();
  }
  function finish() {
    if (S.ended) return;
    S.ended = true;
    S.playing = false;
    var feedRows = S.rows.filter(function (r) { return !r.item.custom && r.data; });
    var labelled = feedRows.filter(function (r) { return r.item.label; });
    var agree = labelled.filter(function (r) { return r.data.decision === r.item.label; }).length;
    var c = { allow: 0, ask: 0, deny: 0 };
    feedRows.forEach(function (r) { if (c[r.data.decision] != null) c[r.data.decision]++; });
    var end = el("div", "aa-end");
    end.id = "aa-end";
    end.appendChild(document.createTextNode("— "));
    end.appendChild(bi("end of session", "会话结束"));
    end.appendChild(document.createTextNode(" · "));
    ["allow", "ask", "deny"].forEach(function (k) {
      var b = el("b", k, VERDICT[k][0] + " " + c[k]);
      end.appendChild(b);
      end.appendChild(document.createTextNode(" "));
    });
    end.appendChild(document.createTextNode("· "));
    end.appendChild(bi("matches the labels ", "与标注一致 "));
    end.appendChild(el("b", null, agree + "/" + labelled.length));
    end.appendChild(document.createTextNode(" —"));
    end.title = t("Labels: hand-written allow / ask / deny in integrations/claude-code-guard/examples",
                  "标注：integrations/claude-code-guard/examples 里手写的允许 / 询问 / 拒绝");
    S.follow = nearBottom();
    $("aa-feed").appendChild(end);
    toBottom();
    renderButtons();
  }
  function checkCustom(cmd) {
    cmd = String(cmd || "").trim();
    if (!cmd) return Promise.resolve(false);
    pause();
    var gen = S.gen;
    $("aa-check").disabled = true;
    return S.current.then(function () {
      if (gen !== S.gen) return false;
      return run({ custom: true, tool: "Bash", input: { command: cmd },
                   cwd: S.ctx.cwd, project_dir: S.ctx.project_dir, home: S.ctx.home });
    }).then(function (ok) {
      $("aa-check").disabled = false;
      if (ok) select(S.rows[S.rows.length - 1], true);
      return ok;
    });
  }
  function setSpeed(v) {
    v = +v;
    if (!(v >= 0.25 && v <= 16)) return;
    S.speed = v;
    Array.prototype.forEach.call($("aa-speed").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", String(+b.getAttribute("data-speed") === v));
    });
  }

  // ------------------------------------------------------------------ rendering
  function renderButtons() {
    var b = $("aa-play"), st = S.playing ? "playing" : S.ended ? "ended" : "paused";
    if (b.getAttribute("data-state") !== st) {        // re-render only on change, so a click is never lost
      b.innerHTML = "";
      b.appendChild(st === "playing" ? bi("❚❚ Pause", "❚❚ 暂停") : st === "ended" ? bi("↻ Replay", "↻ 重播") : bi("▶ Play", "▶ 播放"));
      b.setAttribute("data-state", st);
    }
    b.disabled = !S.items.length;
    $("aa-step").disabled = S.playing || S.busy || S.idx >= S.items.length;
    $("aa-reset").disabled = !S.rows.length && !S.idx;
  }
  function renderStats() {
    var avg = mean(S.ms);
    $("aa-ms").textContent = fmtMs(avg);
    $("aa-model").textContent = S.model || "–";
    $("aa-model").title = S.model || "";
    ["allow", "ask", "deny"].forEach(function (k) { $("aa-n-" + k).textContent = S.counts[k]; });
  }
  function select(entry, byUser) {
    if (!entry) return;
    if (byUser) S.pinned = entry !== S.rows[S.rows.length - 1] || entry.item.custom;
    if (S.selected) S.selected.row.setAttribute("aria-current", "false");
    S.selected = entry;
    entry.row.setAttribute("aria-current", "true");
    renderSent();
  }
  function renderSent() {
    var e = S.selected, head = $("aa-sent-head"), vt = $("aa-vt");
    head.innerHTML = ""; vt.innerHTML = "";
    if (!e || !e.data) {
      head.textContent = "–";
      ["aa-rule", "aa-req", "aa-res", "aa-mreq", "aa-mres"].forEach(function (id) { $(id).textContent = "–"; });
      if (e && e.request) $("aa-req").textContent = json(e.request);
      return;
    }
    var d = e.data, it = e.item;
    var line = el("div", "cmdline", (e.d.bash ? "$ " : e.d.tag + " ") + e.d.text + (e.d.extra ? "  " + e.d.extra : ""));
    head.appendChild(line);
    head.appendChild(chipEl(d.decision, false));
    var s = SOURCE[d.source] || [d.source, d.source];
    var src = el("span", "mj-small");
    src.appendChild(bi("source: ", "来源：")); src.appendChild(bi(s[0], s[1]));
    head.appendChild(src);
    if (d.model_decision) {
      var md = el("span", "mj-small mj-muted");
      md.appendChild(bi("model alone: ", "只看模型：")); md.appendChild(bi(VERDICT[d.model_decision] ? VERDICT[d.model_decision][1] : d.model_decision,
                                                                         VERDICT[d.model_decision] ? VERDICT[d.model_decision][2] : d.model_decision));
      head.appendChild(md);
    }
    var lab = el("span", "mj-small mj-muted");
    if (it.custom) lab.appendChild(bi("your command", "你的命令"));
    else {
      lab.appendChild(bi("label: ", "标注："));
      lab.appendChild(bi(VERDICT[it.label] ? VERDICT[it.label][1] : String(it.label), VERDICT[it.label] ? VERDICT[it.label][2] : String(it.label)));
      lab.appendChild(document.createTextNode(" · " + it.id + (it.note ? " · " + it.note : "")));
    }
    head.appendChild(lab);
    QUESTIONS.forEach(function (q) {
      var tr = el("tr"), v = (d.values || {})[q], th = (d.thresholds || {})[q] || {}, lv = levelOf(d, q);
      var name = el("td"); name.appendChild(bi(Q_NAMES[q][0], Q_NAMES[q][1])); tr.appendChild(name);
      tr.appendChild(el("td", null, v == null ? "–" : q === "risk" ? v.toFixed(2) + " / 4" : v.toFixed(3)));
      tr.appendChild(el("td", null, th.ask == null ? "–" : String(th.ask)));
      tr.appendChild(el("td", null, th.deny == null ? "–" : String(th.deny)));
      var res = el("td", lv ? "lv-" + lv : "mj-muted");
      if (lv) res.appendChild(bi(VERDICT[lv][1], VERDICT[lv][2])); else res.textContent = "–";
      tr.appendChild(res);
      vt.appendChild(tr);
    });
    var rule = $("aa-rule");
    rule.innerHTML = "";
    if (d.rule) {
      rule.appendChild(el("code", null, d.rule.rule));
      rule.appendChild(document.createTextNode(" → "));
      rule.appendChild(bi(VERDICT[d.rule.decision] ? VERDICT[d.rule.decision][1] : d.rule.decision,
                          VERDICT[d.rule.decision] ? VERDICT[d.rule.decision][2] : d.rule.decision));
      rule.appendChild(document.createTextNode(": " + (d.rule.reason || "")));
      if (d.source !== "rule") {
        rule.appendChild(document.createTextNode(" "));
        rule.appendChild(bi("(the model was already at this level)", "（模型已经给出同一级别）"));
      }
    } else rule.appendChild(bi("No hard rule matched.", "没有命中硬规则。"));
    if (d.error) { rule.appendChild(document.createTextNode("  ")); rule.appendChild(bi("error: ", "错误：")); rule.appendChild(document.createTextNode(d.error)); }
    $("aa-req").textContent = json(e.request);
    var rest = {};
    Object.keys(d).forEach(function (k) { if (k !== "call") rest[k] = d[k]; });
    rest.call = "(" + t("below", "见下") + ")";
    $("aa-res").textContent = json(rest);
    var call = d.call || {};
    $("aa-mreq").textContent = call.request ? json(call.request) : "–";
    $("aa-mres").textContent = call.response ? json(call.response) : "–";
  }
  function refreshLanguage() {
    S.rows.forEach(function (e) {
      if (!e.data) return;
      BARS.forEach(function (q, k) { e.bars.children[k].title = barTitle(e.data, q); });
    });
    renderSent();
  }

  // ------------------------------------------------------------------ start-up
  function readUrl() {
    var q = new URLSearchParams(location.search);
    if (q.has("speed")) setSpeed(parseFloat(q.get("speed")));
    return q.get("autoplay") === "1";
  }
  function init() {
    M = window.JevStyle;
    var autoplay = readUrl();
    $("aa-play").addEventListener("click", function () { if (S.playing) pause(); else play(); });
    $("aa-bigplay").addEventListener("click", play);
    $("aa-step").addEventListener("click", function () { step(); });
    $("aa-reset").addEventListener("click", reset);
    Array.prototype.forEach.call($("aa-speed").querySelectorAll("button"), function (b) {
      b.addEventListener("click", function () { setSpeed(b.getAttribute("data-speed")); });
    });
    $("aa-try").addEventListener("submit", function (ev) {
      ev.preventDefault();
      var input = $("aa-cmd"), v = input.value;
      if (!v.trim() || $("aa-check").disabled) return;
      input.value = "";                                   // like a terminal: the line moves into the feed
      checkCustom(v).then(function (ok) { if (!ok && !input.value) input.value = v; });
    });
    $("aa-cmd").addEventListener("input", function () { if (S.playing) pause(); });   // you type, the agent waits
    $("aa-feed").addEventListener("scroll", function () { S.follow = nearBottom(); }, { passive: true });
    document.addEventListener("jevstyle:lang", refreshLanguage);
    renderButtons();
    fetch("/api/agent-approval/feed").then(function (r) {
      if (!r.ok) throw new Error("GET /api/agent-approval/feed -> HTTP " + r.status);
      return r.json();
    }).then(function (feed) {
      S.feed = feed;
      S.items = feed.items || [];
      S.ctx = feed.context || { cwd: "/work/shop-api", project_dir: "/work/shop-api", home: "/home/dev" };
      S.model = feed.engine && (feed.engine.model_name || feed.engine.model);
      $("aa-session").textContent = "agent · " + S.ctx.cwd.replace(S.ctx.home, "~");
      renderStats();
      renderButtons();
      document.documentElement.setAttribute("data-aa-ready", "");
      if (autoplay) play();
    }).catch(function (e) {
      showError(t("Could not load the session: ", "会话加载失败：") + (e && e.message || e));
      document.documentElement.setAttribute("data-aa-ready", "error");
    });
  }

  window.AgentApproval = {
    state: S, play: play, pause: pause, step: step, reset: reset, check: checkCustom, setSpeed: setSpeed,
    whenIdle: function () { return S.current; }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
