/* Snake demo UI. Game rules and the request live in snake-core.js; this file draws, runs the loop and
 * calls POST /v1/systemone through JevStyle.decide (common.js). No build step, no third-party code.
 * URL parameters (optional, kept in sync): seed, size, player (model|greedy), filter (0|1), speed (1..21, 21 = max),
 * wording (v1 = default: the model only hears whether each move gets closer to the food; v0 = the first version,
 * which also sent the free-cell count; kept for comparisons, no visible control).
 * Test hook: window.SnakeDemo; <html data-snake-ready> once initialised. */
(function () {
  "use strict";
  var M, C, $ = function (id) { return document.getElementById(id); };
  var ARROWS = { up: "↑", down: "↓", left: "←", right: "→" };
  var NAMES_ZH = { up: "上", down: "下", left: "左", right: "右" };
  var MAX_SPEED = 21;
  var S = {
    game: null, view: null, running: false, busy: false, gen: 0, loopId: 0,
    player: "model", filter: true, seed: 1, size: 10, speed: 6, auto: false, wording: null,
    ms: [], serverMs: null, moveTimes: [], model: null,
    session: { model: [], greedy: [] },
    cmp: { running: false, stop: false, rows: [], model: null, game: null }
  };

  function t(en, zh) { return M.t(en, zh); }
  function now() { return (window.performance || Date).now(); }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function moveName(m) { return t(m, NAMES_ZH[m]); }
  function reasonText(r) {
    return ({ wall: t("hit the wall", "撞墙"), body: t("hit itself", "撞到自己"),
              stalled: t("no food for too long", "太久没吃到食物"), full: t("board full", "占满棋盘"),
              error: t("stopped by an error", "因错误停止") })[r] || r || "";
  }
  function fmtMs(ms) { return ms == null ? "–" : ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : (ms < 10 ? ms.toFixed(1) : Math.round(ms)) + ""; }
  function mean(xs) { return xs.length ? xs.reduce(function (a, b) { return a + b; }, 0) / xs.length : null; }
  function pct(p) { return (p * 100).toFixed(1) + "%"; }

  // ------------------------------------------------------------------ settings <-> URL
  function readUrl() {
    var q = new URLSearchParams(location.search);
    if (q.has("seed")) S.seed = C.normSeed(q.get("seed"));
    var size = parseInt(q.get("size"), 10);
    if (C.GRIDS.indexOf(size) >= 0) S.size = size;
    if (q.get("player") === "greedy" || q.get("player") === "model") S.player = q.get("player");
    if (q.has("filter")) S.filter = q.get("filter") !== "0";
    var sp = parseInt(q.get("speed"), 10);
    if (sp >= 1 && sp <= MAX_SPEED) S.speed = sp;
    var wd = q.get("wording");
    if (wd && C.WORDINGS[wd]) S.wording = wd;
  }
  function writeUrl() {
    try {
      var q = new URLSearchParams(location.search);
      q.set("seed", S.seed); q.set("size", S.size); q.set("player", S.player);
      q.set("filter", S.filter ? "1" : "0"); q.set("speed", S.speed);
      if (S.wording !== C.DEFAULT_WORDING) q.set("wording", S.wording); else q.delete("wording");
      history.replaceState(null, "", location.pathname + "?" + q.toString());
    } catch (e) { /* file:// or sandboxed: ignore */ }
  }

  // ------------------------------------------------------------------ drawing
  function colors() {
    var cs = getComputedStyle(document.documentElement);
    function v(n, d) { var x = cs.getPropertyValue(n).trim(); return x || d; }
    return { card: v("--card", "#fff"), soft: v("--soft", "#eee"), line: v("--line", "#ddd"), fg: v("--fg", "#111"),
             accent: v("--accent", "#2f5bd3"), accentFg: v("--accent-fg", "#fff"), bar: v("--bar", "#2f5bd3"),
             block: v("--block", "#b3261e"), act: v("--act", "#1f7a3f"), muted: v("--muted", "#777") };
  }
  function hex(c) {
    var m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(c || "");
    if (!m) return null;
    var h = m[1].length === 3 ? m[1].replace(/./g, "$&$&") : m[1];
    return [0, 2, 4].map(function (i) { return parseInt(h.substr(i, 2), 16); });
  }
  // solid mix of two #hex colours (no alpha, so overlapping body segments do not darken)
  function mix(a, b, k) {
    var x = hex(a), y = hex(b);
    if (!x || !y) return a;
    return "rgb(" + x.map(function (v, i) { return Math.round(v + (y[i] - v) * k); }).join(",") + ")";
  }
  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  function draw(g) {
    g = g || S.game;
    var cv = $("sn-board"), wrap = $("sn-wrap");
    var css = Math.max(40, Math.floor(wrap.clientWidth));
    var dpr = window.devicePixelRatio || 1, px = Math.round(css * dpr);
    if (cv.width !== px || cv.height !== px) { cv.width = px; cv.height = px; }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var col = colors(), n = g.size, c = css / n, i;
    ctx.fillStyle = col.card;
    ctx.fillRect(0, 0, css, css);
    ctx.fillStyle = col.soft;
    for (var y = 0; y < n; y++) for (var x = 0; x < n; x++) if ((x + y) % 2) ctx.fillRect(x * c, y * c, c, c);
    if (g.food) {
      ctx.fillStyle = col.block;
      ctx.beginPath();
      ctx.arc((g.food[0] + 0.5) * c, (g.food[1] + 0.5) * c, c * 0.32, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = col.act;
      ctx.fillRect((g.food[0] + 0.5) * c - c * 0.04, g.food[1] * c + c * 0.1, c * 0.08, c * 0.16);
    }
    var len = g.snake.length, pad = c * 0.08;
    for (i = len - 1; i >= 1; i--) {
      var p = g.snake[i], q = g.snake[i - 1];
      ctx.fillStyle = mix(col.bar, col.card, 0.5 * (i / Math.max(1, len - 1)));
      // join each segment to the one before it so the body reads as one line
      var x0 = Math.min(p[0], q[0]), y0 = Math.min(p[1], q[1]);
      var w = (Math.abs(p[0] - q[0]) + 1) * c, h = (Math.abs(p[1] - q[1]) + 1) * c;
      roundRect(ctx, x0 * c + pad, y0 * c + pad, w - 2 * pad, h - 2 * pad, c * 0.28);
      ctx.fill();
    }
    var hd = g.snake[0];
    ctx.fillStyle = col.accent;
    roundRect(ctx, hd[0] * c + pad * 0.5, hd[1] * c + pad * 0.5, c - pad, c - pad, c * 0.3);
    ctx.fill();
    var d = C.DIRS[g.dir], ex = (hd[0] + 0.5) * c, ey = (hd[1] + 0.5) * c;
    var fx = d[0] * c * 0.18, fy = d[1] * c * 0.18, sx = -d[1] * c * 0.17, sy = d[0] * c * 0.17;
    ctx.fillStyle = col.accentFg;
    [1, -1].forEach(function (s) {
      ctx.beginPath();
      ctx.arc(ex + fx + s * sx, ey + fy + s * sy, Math.max(1.2, c * 0.08), 0, Math.PI * 2);
      ctx.fill();
    });
    if (g.crash) {
      ctx.strokeStyle = col.block;
      ctx.lineWidth = Math.max(2, c * 0.1);
      ctx.lineCap = "round";
      var cx = g.crash[0], cy = g.crash[1];
      if (cx >= 0 && cy >= 0 && cx < n && cy < n) {
        ctx.beginPath();
        ctx.moveTo(cx * c + c * 0.25, cy * c + c * 0.25); ctx.lineTo(cx * c + c * 0.75, cy * c + c * 0.75);
        ctx.moveTo(cx * c + c * 0.75, cy * c + c * 0.25); ctx.lineTo(cx * c + c * 0.25, cy * c + c * 0.75);
        ctx.stroke();
      } else {                                  // wall: a red bar on the side of the head it ran into
        var bx = (hd[0] + 0.5 + d[0] * 0.5) * c, by = (hd[1] + 0.5 + d[1] * 0.5) * c;
        ctx.beginPath();
        ctx.moveTo(bx - d[1] * c * 0.4, by - d[0] * c * 0.4);
        ctx.lineTo(bx + d[1] * c * 0.4, by + d[0] * c * 0.4);
        ctx.stroke();
      }
    }
  }

  // ------------------------------------------------------------------ decisions
  function errorText(r) {
    var e = (r.data && r.data.error) || {};
    if (r.status === 401) return t("The server needs an API key: set it in the Playground.", "服务需要 API key，请先在试用页填写。");
    if (r.status === 0) return t("Server not reachable: ", "连不上服务：") + (e.message || "");
    return "HTTP " + r.status + ": " + (e.code ? e.code + " – " : "") + (e.message || "");
  }
  /* One decision for game g. Returns {dec, player, source, choice, resp, ms, error}.
     source: "model" | "greedy" | "forced" | "no_safe". Code-only moves never call the server. */
  function decideMove(g, player, filter) {
    var dec = C.buildDecision(g, { filter: filter, wording: S.wording });
    var out = { dec: dec, player: player, source: null, choice: null, resp: null, ms: null, error: null };
    if (player === "greedy") { out.source = "greedy"; out.choice = C.greedyChoice(dec); return Promise.resolve(out); }
    var cm = C.codeMove(dec);
    if (cm) { out.source = dec.mode; out.choice = cm; return Promise.resolve(out); }
    return M.decide(dec.request).then(function (r) {
      out.resp = r.data;
      out.ms = r.ms;
      if (!r.ok) { out.error = errorText(r); return out; }
      var bad = C.checkResponse(r.data, dec);
      if (bad.length) { out.error = t("Unexpected response: ", "响应不符合约定：") + bad.join("; "); return out; }
      out.source = "model";
      out.choice = r.data.answers[C.QUESTION_ID].choice;
      return out;
    });
  }

  function tick() {
    if (S.busy || S.game.over || S.cmp.running) return Promise.resolve(false);
    S.busy = true;
    var gen = S.gen;
    return decideMove(S.game, S.player, S.filter).then(function (d) {
      if (gen !== S.gen) return false;                  // reset while the request was in flight
      S.busy = false;
      S.view = d;
      if (d.error) { showError(d.error); setRunning(false); renderPanel(); return false; }
      showError(null);
      C.applyMove(S.game, d.choice);
      if (d.source === "model") {
        S.ms.push(d.ms);
        if (S.ms.length > 20) S.ms.shift();
        S.serverMs = d.resp.timing && d.resp.timing.total_ms;
        S.model = d.resp.model;
      }
      if (S.running) {                                  // moves/s is measured on the running loop only
        S.moveTimes.push(now());
        if (S.moveTimes.length > 20) S.moveTimes.shift();
      }
      if (S.game.over) finishGame();
      render();
      return true;
    }).catch(function (e) {
      if (gen === S.gen) { S.busy = false; showError(String(e && e.message || e)); setRunning(false); }
      return false;
    });
  }

  function finishGame() {
    var g = S.game;
    S.session[S.player].push({ seed: g.seed, score: g.score, steps: g.steps, reason: g.reason });
  }

  function loop(id) {
    if (!S.running || id !== S.loopId) return;
    var t0 = now();
    tick().then(function (ok) {
      if (!S.running || id !== S.loopId) return;
      if (S.game.over) {
        if (S.auto) {
          sleep(700).then(function () {
            if (!S.running || id !== S.loopId) return;
            setSeed(S.seed + 1);
            newGame();
            loop(id);
          });
        } else setRunning(false);
        return;
      }
      if (!ok) { setRunning(false); return; }
      var interval = S.speed >= MAX_SPEED ? 0 : 1000 / S.speed;
      sleep(Math.max(0, interval - (now() - t0))).then(function () { loop(id); });
    });
  }
  function setRunning(on) {
    S.running = !!on;
    if (on) { S.loopId++; S.moveTimes = []; loop(S.loopId); }
    renderButtons();
  }
  function newGame() {
    S.gen++;
    S.busy = false;
    S.game = C.newGame({ size: S.size, seed: S.seed });
    S.view = null;
    showError(null);
    render();
  }
  function setSeed(v) {
    S.seed = C.normSeed(v);
    $("sn-seed").value = S.seed;
    writeUrl();
  }

  // ------------------------------------------------------------------ rendering
  function showError(msg) { var e = $("sn-error"); e.hidden = !msg; e.textContent = msg || ""; }
  function renderButtons() {
    var b = $("sn-start");
    b.innerHTML = "";
    b.appendChild(S.running ? M.bi("Pause", "暂停") : S.game && S.game.over ? M.bi("Next game", "下一局") : M.bi("Start", "开始"));
    var cmp = S.cmp.running;
    b.disabled = cmp;
    $("sn-step").disabled = cmp || S.running || S.game.over;
    $("sn-reset").disabled = cmp;
    $("sn-cmp-run").disabled = cmp;
    $("sn-cmp-stop").hidden = !cmp;
    ["sn-seed", "sn-size", "sn-dice"].forEach(function (id) { $(id).disabled = cmp; });
  }
  function renderStats() {
    var g = S.game;
    $("sn-score").textContent = g.score;
    $("sn-length").textContent = g.snake.length;
    $("sn-steps").textContent = g.steps;
    var avg = mean(S.ms);
    $("sn-ms").textContent = S.player === "greedy" || avg == null ? "–" : avg >= 1000 ? fmtMs(avg) : fmtMs(avg) + " ms";
    $("sn-ms-sub").textContent = S.player === "greedy" ? t("code only", "只用代码")
      : S.serverMs != null ? t("server ", "服务端 ") + fmtMs(S.serverMs) + (S.serverMs >= 1000 ? "" : " ms") : "";
    var mt = S.moveTimes, mps = mt.length >= 2 ? (mt.length - 1) * 1000 / Math.max(1, mt[mt.length - 1] - mt[0]) : null;
    $("sn-mps").textContent = mps == null ? "–" : mps >= 100 ? Math.round(mps) : mps.toFixed(1);
    var over = $("sn-over");
    if (g.over && !S.cmp.running) {
      over.hidden = false;
      over.innerHTML = "";
      over.appendChild(document.createTextNode(t("Game over: ", "游戏结束：") + reasonText(g.reason)));
      over.appendChild(el("span", "sub", t("score ", "得分 ") + g.score + " · " + t("seed ", "种子 ") + g.seed));
    } else over.hidden = true;
    var parts = [];
    ["model", "greedy"].forEach(function (p) {
      var xs = S.session[p];
      if (!xs.length) return;
      var best = Math.max.apply(null, xs.map(function (r) { return r.score; }));
      parts.push((p === "model" ? "Jev-Style v3" : t("greedy", "贪心")) + ": " + xs.length + t(" games, best ", " 局，最高 ") + best);
    });
    $("sn-session").textContent = parts.length ? t("Finished — ", "已完成 — ") + parts.join(" · ") : "";
  }

  function shortFact(f, status, dec) {
    if (status === "reverse") return t("reverse: not offered", "反向：不提供");
    if (f.deadly) {
      var why = f.cause === "wall" ? t("wall", "撞墙") : t("own body", "撞到自己");
      return status === "filtered" ? why + t(" · filtered by code", " · 已被代码过滤") : why + t(" · deadly, offered", " · 致命，但仍提供");
    }
    // what the model hears (v1) first, then the count only code uses
    var s = f.eats ? t("eats the food", "吃到食物") :
      (f.distAfter < f.distBefore ? t("closer ", "靠近食物 ") : t("away ", "远离食物 ")) + f.distBefore + "→" + f.distAfter;
    s += " · " + t(f.reachable + " free cells", "可达空格 " + f.reachable);
    if (f.reachable < f.length) s += t(", fewer than length", "，少于蛇长");
    if (dec.wording !== "v0") s += t(" (code only)", "（仅代码用）");
    return s;
  }
  function statusOf(dec, m) {
    if (dec.excluded[m] === "reverse") return "reverse";
    if (dec.excluded[m] === "deadly") return "filtered";
    return "offered";
  }
  function renderPanel() {
    var box = $("sn-moves"), d = S.view, dec = d ? d.dec : C.buildDecision(S.game, { filter: S.filter, wording: S.wording });
    var probs = d && d.source === "model" ? d.resp.answers[C.QUESTION_ID].probabilities : null;
    box.innerHTML = "";
    C.ORDER.forEach(function (m) {
      var f = dec.facts[m], st = statusOf(dec, m);
      var row = el("div", "sn-mv");
      row.setAttribute("data-move", m);
      row.setAttribute("data-status", st);
      if (f.deadly && st !== "reverse") row.setAttribute("data-deadly", "");
      if (d && d.choice === m) row.setAttribute("data-chosen", "");
      row.appendChild(el("span", "arrow", ARROWS[m]));
      row.appendChild(el("span", "name", moveName(m)));
      var trk = el("div", "trk"), fil = el("div", "fil");
      var p = probs && m in probs ? probs[m] : null;
      if (p == null && d && d.choice === m && d.source !== "model") p = 1;
      fil.style.width = ((p || 0) * 100).toFixed(2) + "%";
      trk.appendChild(fil);
      row.appendChild(trk);
      var val = probs && m in probs ? pct(probs[m]) : d && d.choice === m ? t("code", "代码") : "–";
      row.appendChild(el("span", "val", val));
      row.appendChild(el("span", "fact", shortFact(f, st, dec)));
      box.appendChild(row);
    });
    var mode = $("sn-mode"), conf = $("sn-conf");
    mode.className = "mj-chip";
    conf.textContent = "";
    if (!d) mode.textContent = t("next: ", "下一步：") + (S.player === "greedy" ? t("greedy code bot", "贪心代码") :
      dec.mode === "ask" ? t("Jev-Style v3 picks among ", "Jev-Style v3 从 ") + dec.offered.length + t(" moves", " 个方向里选") :
      dec.mode === "forced" ? t("one safe move", "只有一个安全方向") : t("no safe move", "没有安全方向"));
    else if (d.error) { mode.textContent = t("error", "出错"); mode.className = "mj-chip block"; }
    else if (d.source === "model") {
      mode.textContent = t("Jev-Style v3 chose ", "Jev-Style v3 选了 ") + moveName(d.choice);
      mode.className = "mj-chip act";
      conf.textContent = t("confidence ", "置信度 ") + d.resp.answers[C.QUESTION_ID].confidence.toFixed(2);
    } else if (d.source === "greedy") mode.textContent = t("greedy code bot chose ", "贪心代码选了 ") + moveName(d.choice);
    else if (d.source === "forced") { mode.textContent = t("forced: only one safe move, model not asked", "只剩一个安全方向，直接走，没问模型"); mode.className = "mj-chip review"; }
    else { mode.textContent = t("no safe move left", "已无安全方向"); mode.className = "mj-chip block"; }
    $("sn-honest").textContent = S.player === "greedy"
      ? t("Greedy code bot: code alone picks the safe move closest to the food (ties: more free cells). No model call.",
          "贪心代码：只用代码，在安全方向里选离食物最近的（平手时选可达空格多的），不调用模型。")
      : S.wording === "v0"
        ? t("Wording v0 (from the URL): Jev-Style v3 is sent every code fact, including the free-cell count" +
            (S.filter ? "; code still removes deadly moves." : "; deadly moves are offered too."),
            "措辞 v0（来自网址）：代码算出的全部事实都发给 Jev-Style v3，包括可达空格数" +
            (S.filter ? "；致命方向仍由代码去掉。" : "；致命方向也会给它。"))
      : S.filter
        ? t("Jev-Style v3 sees only whether each move gets closer to the food (distance before→after). " +
            "Code alone removes deadly moves and counts free cells; that count is shown here, not sent.",
            "Jev-Style v3 只看到每个方向是靠近还是远离食物（前后距离）。致命方向由代码单独去掉，" +
            "可达空格也只由代码计算，只在这里显示，不发给模型。")
        : t("Filter off: Jev-Style v3 sees whether each move gets closer to the food, and deadly moves are offered " +
            "too, labelled \u201cdeadly\u201d. Free cells are counted by code and shown here, not sent.",
            "过滤已关：Jev-Style v3 看到每个方向是靠近还是远离食物，致命方向也会给它，并标明 \u201cdeadly\u201d。" +
            "可达空格只由代码计算，只在这里显示，不发给模型。");
    renderDecision(dec, d, probs);
  }

  function renderDecision(dec, d, probs) {
    var tb = $("sn-facts");
    tb.innerHTML = "";
    C.ORDER.forEach(function (m) {
      var f = dec.facts[m], st = statusOf(dec, m), tr = el("tr");
      tr.appendChild(el("td", null, ARROWS[m] + " " + m));
      tr.appendChild(el("td", null, st === "reverse" ? t("reverse, not offered", "反向，不提供") :
        st === "filtered" ? t("deadly, removed", "致命，已去掉") : dec.mode === "ask" ? t("offered", "提供") :
        dec.mode === "forced" ? t("only safe move", "唯一安全方向") : t("—", "—")));
      var sent = st === "reverse" ? "" : st === "filtered" ? t("(not sent)", "（未发送）") :
        dec.mode === "ask" ? C.optionText(f, dec.wording) : t("(not sent: code moves)", "（未发送：代码直接走）");
      tr.appendChild(el("td", st === "offered" && dec.mode === "ask" ? "sent" : "muted", sent));
      tr.appendChild(el("td", "code", st === "reverse" ? "" : C.factText(f)));
      tr.appendChild(el("td", null, probs && m in probs ? probs[m].toFixed(4) : ""));
      tb.appendChild(tr);
    });
    var req = $("sn-req"), res = $("sn-res");
    $("sn-statetext").textContent = dec.request ? dec.request.state : "–";
    if (S.player === "greedy" && (!d || d.source === "greedy")) {
      req.textContent = t("Nothing sent: the greedy code bot does not call the model.", "未发送：贪心代码不调用模型。");
      res.textContent = "–";
    } else if (dec.mode !== "ask") {
      req.textContent = dec.mode === "forced"
        ? t("Nothing sent: only one safe move, code takes it.", "未发送：只有一个安全方向，代码直接走。")
        : t("Nothing sent: every move is deadly, the game ends.", "未发送：每个方向都会死，游戏结束。");
      res.textContent = "–";
    } else {
      req.textContent = JSON.stringify(dec.request, null, 2);
      res.textContent = d && d.resp ? JSON.stringify(d.resp, null, 2) : "–";
    }
  }

  function render() {
    draw();
    renderStats();
    renderPanel();
    renderButtons();
  }

  // ------------------------------------------------------------------ compare
  function cmpCell(r) {
    var td = el("td");
    if (!r) { td.textContent = "…"; return td; }
    td.appendChild(el("span", "sc", String(r.score)));
    var extra = reasonText(r.reason) + " · " + r.steps + t(" steps", " 步");
    if (r.ms != null) extra += " · " + fmtMs(r.ms) + " ms";
    td.appendChild(el("span", "end", extra));
    return td;
  }
  function renderCompare() {
    var body = $("sn-cmp-body"), foot = $("sn-cmp-foot");
    body.innerHTML = "";
    foot.innerHTML = "";
    S.cmp.rows.forEach(function (row, i) {
      var tr = el("tr");
      if (row.model && row.greedy && row.model.score > row.greedy.score) tr.className = "better";
      tr.appendChild(el("td", null, String(i + 1)));
      tr.appendChild(el("td", null, String(row.seed)));
      tr.appendChild(cmpCell(row.model));
      tr.appendChild(cmpCell(row.greedy));
      body.appendChild(tr);
    });
    var done = S.cmp.rows.filter(function (r) { return r.model && r.greedy; });
    if (!done.length) return;
    var mm = mean(done.map(function (r) { return r.model.score; })), gm = mean(done.map(function (r) { return r.greedy.score; }));
    var ahead = done.filter(function (r) { return r.model.score >= r.greedy.score; }).length;
    var ms = mean(done.filter(function (r) { return r.model.ms != null; }).map(function (r) { return r.model.ms; }));
    var tr = el("tr");
    tr.appendChild(el("td", null, ""));
    tr.appendChild(el("td", null, t("mean", "平均")));
    var a = el("td"), b = el("td");
    a.appendChild(el("span", "sc", mm.toFixed(1)));
    a.appendChild(el("span", "end", t("≥ bot in ", "不输给代码：") + ahead + "/" + done.length +
                     (ms != null ? " · " + fmtMs(ms) + " ms" : "") + (S.cmp.model ? " · " + S.cmp.model : "")));
    b.appendChild(el("span", "sc", gm.toFixed(1)));
    b.appendChild(el("span", "end", t("code only", "只用代码")));
    tr.appendChild(a);
    tr.appendChild(b);
    foot.appendChild(tr);
  }
  function cmpStatus(msg) { $("sn-cmp-status").textContent = msg || ""; }

  function playCompareGame(seed, player, filter, size, label) {
    var g = C.newGame({ size: size, seed: seed }), ms = [], badge = $("sn-badge");
    S.cmp.game = g;
    function step() {
      if (S.cmp.stop) return Promise.resolve(null);
      if (g.over) return Promise.resolve({ score: g.score, steps: g.steps, reason: g.reason, ms: mean(ms) });
      return decideMove(g, player, filter).then(function (d) {
        if (d.error) throw new Error(d.error);
        C.applyMove(g, d.choice);
        if (d.ms != null) { ms.push(d.ms); S.cmp.model = d.resp.model; }
        badge.textContent = label + " · " + t("step ", "第 ") + g.steps + t("", " 步") + " · " + t("score ", "得分 ") + g.score;
        // greedy games need no network: draw every step but yield to the browser now and then
        if (player === "greedy" && g.steps % 4) return step();
        draw(g);
        return (player === "greedy" ? sleep(0) : Promise.resolve()).then(step);
      });
    }
    return step();
  }
  function runCompare() {
    var n = Math.max(1, Math.min(50, parseInt($("sn-cmp-n").value, 10) || 5));
    $("sn-cmp-n").value = n;
    setRunning(false);
    S.gen++;
    S.busy = false;
    S.cmp = { running: true, stop: false, rows: [], model: null, game: null };
    var base = S.seed, filter = S.filter, size = S.size, badge = $("sn-badge");
    badge.hidden = false;
    $("sn-over").hidden = true;
    renderButtons();
    renderCompare();
    showError(null);
    var i = 0;
    function next() {
      if (S.cmp.stop || i >= n) return Promise.resolve();
      var row = { seed: C.normSeed(base + i) };
      S.cmp.rows.push(row);
      renderCompare();
      var k = i + 1;
      cmpStatus(t("game ", "第 ") + k + "/" + n + t("", " 局"));
      return playCompareGame(row.seed, "model", filter, size, "#" + k + " Jev-Style v3").then(function (r) {
        if (!r) return;
        row.model = r;
        renderCompare();
        return playCompareGame(row.seed, "greedy", filter, size, "#" + k + " " + t("greedy", "贪心")).then(function (r2) {
          if (!r2) return;
          row.greedy = r2;
          renderCompare();
          i++;
          return next();
        });
      });
    }
    return next().then(function () {
      cmpStatus((S.cmp.stop ? t("stopped", "已停止") : t("done: ", "完成：") + n + t(" games each", " 局 × 2")) +
                (filter ? t(" · deadly moves filtered", " · 已过滤致命方向") : t(" · filter off", " · 未过滤")));
    }, function (e) {
      showError(e.message || String(e));
      cmpStatus(t("stopped by an error", "因错误停止"));
    }).then(function () {
      S.cmp.rows = S.cmp.rows.filter(function (r) { return r.model && r.greedy; });
      S.cmp.running = false;
      badge.hidden = true;
      renderCompare();
      render();
    });
  }

  // ------------------------------------------------------------------ wiring
  function syncControls() {
    $("sn-seed").value = S.seed;
    $("sn-size").value = String(S.size);
    $("sn-filter").checked = S.filter;
    $("sn-auto").checked = S.auto;
    $("sn-speed").value = S.speed;
    $("sn-speed-v").textContent = S.speed >= MAX_SPEED ? t("max", "最快") : S.speed + "/s";
    var btns = $("sn-player").querySelectorAll("button");
    for (var i = 0; i < btns.length; i++) btns[i].setAttribute("aria-pressed", btns[i].getAttribute("data-player") === S.player ? "true" : "false");
  }
  function init() {
    M = window.JevStyle;
    C = window.SnakeCore;
    if (!M || !C) { document.body.appendChild(el("p", "mj-err", "JevStyle / SnakeCore scripts missing")); return; }
    var size = $("sn-size");
    C.GRIDS.forEach(function (n) { var o = el("option", null, n + " × " + n); o.value = String(n); size.appendChild(o); });
    readUrl();
    if (!S.wording) S.wording = C.DEFAULT_WORDING;
    syncControls();
    S.game = C.newGame({ size: S.size, seed: S.seed });

    $("sn-start").addEventListener("click", function () {
      if (S.running) { setRunning(false); return; }
      if (S.game.over) { setSeed(S.seed + 1); newGame(); }     // Reset replays the same seed
      setRunning(true);
    });
    $("sn-step").addEventListener("click", function () { if (!S.running) tick(); });
    $("sn-reset").addEventListener("click", function () { setRunning(false); newGame(); });
    $("sn-speed").addEventListener("input", function () {
      S.speed = Math.max(1, Math.min(MAX_SPEED, parseInt(this.value, 10) || 6));
      syncControls();
      writeUrl();
    });
    $("sn-seed").addEventListener("change", function () { setRunning(false); setSeed(this.value); newGame(); });
    $("sn-dice").addEventListener("click", function () {
      setRunning(false);
      var r = 0;
      try { r = crypto.getRandomValues(new Uint32Array(1))[0] % 100000; } catch (e) { r = Math.floor(Math.random() * 100000); }
      setSeed(r);
      newGame();
    });
    size.addEventListener("change", function () {
      S.size = parseInt(this.value, 10);
      setRunning(false);
      writeUrl();
      newGame();
    });
    $("sn-filter").addEventListener("change", function () { S.filter = this.checked; writeUrl(); render(); });
    $("sn-auto").addEventListener("change", function () { S.auto = this.checked; });
    $("sn-player").addEventListener("click", function (e) {
      var b = e.target.closest("button[data-player]");
      if (!b) return;
      S.player = b.getAttribute("data-player");
      S.ms = [];
      S.serverMs = null;
      syncControls();
      writeUrl();
      render();
    });
    $("sn-cmp-run").addEventListener("click", function () { runCompare(); });
    $("sn-cmp-stop").addEventListener("click", function () { S.cmp.stop = true; });

    document.addEventListener("jevstyle:lang", function () { syncControls(); render(); renderCompare(); });
    var redraw = function () { draw(S.cmp.running && S.cmp.game ? S.cmp.game : S.game); };
    try { new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] }); } catch (e) {}
    try { window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw); } catch (e) {}
    if (window.ResizeObserver) new ResizeObserver(redraw).observe($("sn-wrap"));
    else window.addEventListener("resize", redraw);

    render();
    window.SnakeDemo = { state: S, core: C, tick: tick, start: function () { if (!S.running) setRunning(true); },
                         pause: function () { setRunning(false); }, reset: function () { setRunning(false); newGame(); },
                         setSeed: function (v) { setSeed(v); newGame(); }, runCompare: runCompare };
    document.documentElement.setAttribute("data-snake-ready", "1");
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
