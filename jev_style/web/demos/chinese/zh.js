/* Chinese & multilingual demo: ticket triage, inbox sorting, 51-language intent. Plain JS. */
(function () {
  "use strict";
  var M = window.JevStyle, el = M.el, bi = M.bi, t = M.t;
  var CASES = null, MASSIVE = null, TRI = 0, TRI_RES = null, INBOX = {}, MAIL_SEL = null, ALL = {}, STOP = false;
  var FOLDER_EN = { "技术问题": "Technical", "账单付款": "Billing", "退换货": "Returns", "销售线索": "Sales leads",
                    "人事": "HR", "客服咨询": "Service & questions" };
  var QTITLE = { queue: ["Queue", "分派队列"], urgency: ["Urgency", "紧急程度"], mood: ["Customer mood", "客户情绪"],
                 human: ["Needs a person", "需要人工"], refund: ["Asks for money back", "要求退款/赔偿"] };

  function $(id) { return document.getElementById(id); }
  function ms(n) { return n < 10 ? n.toFixed(1) + " ms" : Math.round(n).toLocaleString("en-US") + " ms"; }
  function chip(cls, en, zh) { var c = el("span", { "class": "mj-chip " + (cls || "") }); c.appendChild(bi(en, zh)); return c; }
  function status(en, zh, err) {
    var s = $("zh-status"); s.innerHTML = ""; if (en) s.appendChild(bi(en, zh)); s.classList.toggle("mj-err", !!err);
  }
  function fail(r) {
    if (r.status === 401) $("zh-key").hidden = false;
    var e = (r.data && r.data.error) || {};
    status("Error " + r.status + ": " + (e.message || e.code || ""), "出错 " + r.status + "：" + (e.message || e.code || ""), true);
  }
  function argmax(o) { var b = null; Object.keys(o).forEach(function (k) { if (b == null || o[k] > o[b]) b = k; }); return b; }

  // ---- triage -------------------------------------------------------------------------------
  function queueLabels() {
    var out = {};
    Object.keys(CASES.triage.questions.queue.criteria).forEach(function (k) {
      out[k] = t(k, (CASES.queue_zh[k] || k));
    });
    return out;
  }
  function renderTriList() {
    var list = $("tri-list"); list.innerHTML = "";
    CASES.triage.cases.forEach(function (c, i) {
      var b = el("button", { "class": "zh-item", type: "button", role: "option", "aria-selected": i === TRI ? "true" : "false" });
      b.appendChild(el("span", { "class": "s", lang: "zh-CN" }, c.state.subject || c.state.body.slice(0, 24)));
      var ch = el("span", { "class": "zh-chips" });
      ch.appendChild(chip("", c.gold_queue, CASES.queue_zh[c.gold_queue]));
      b.appendChild(ch);
      b.addEventListener("click", function () { TRI = i; TRI_RES = null; loadTri(); });
      list.appendChild(b);
    });
  }
  function loadTri() {
    var c = CASES.triage.cases[TRI];
    $("tri-subject").value = c.state.subject || "";
    $("tri-body").value = c.state.body;
    $("tri-time").textContent = "";
    renderTriList(); renderTriOut();
  }
  function triState() {
    var s = { subject: $("tri-subject").value.trim(), body: $("tri-body").value.trim() };
    return s;
  }
  function ansCard(id, a, extra) {
    var q = CASES.triage.questions[id] || CASES.email.questions[id], box = el("div", { "class": "zh-ans" });
    var h = el("h3"); h.appendChild(bi((QTITLE[id] || [id])[0], (QTITLE[id] || [id, id])[1])); box.appendChild(h);
    var inner = el("div");
    if (a.type === "noul") {
      var yes = a.noul >= 0.5;
      box.appendChild(el("div", { "class": "big" })).appendChild(bi(yes ? "Yes" : "No", yes ? "是" : "否"));
      M.bars(inner, { yes: a.noul, no: 1 - a.noul }, { labels: { yes: t("yes", "是"), no: t("no", "否") } });
    } else if (a.type === "choice") {
      var labels = id === "queue" ? queueLabels() : null;
      box.appendChild(el("div", { "class": "big" }, labels ? labels[a.choice] : a.choice));
      M.bars(inner, a.probabilities, { labels: labels || undefined, order: Object.keys(q.criteria) });
    } else {
      var top = argmax(a.probabilities);
      box.appendChild(el("div", { "class": "big" }, a.legend[top]));
      M.bars(inner, a.probabilities, { labels: a.legend, order: Object.keys(a.legend) });
    }
    (extra || []).forEach(function (x) { h.appendChild(x); });
    box.appendChild(inner);
    return box;
  }
  function renderTriOut() {
    var out = $("tri-out"); out.innerHTML = "";
    if (!TRI_RES) return;
    var c = CASES.triage.cases[TRI], edited = TRI_RES.edited;
    Object.keys(CASES.triage.questions).forEach(function (id) {
      var a = TRI_RES.data.answers[id], extra = [];
      if (id === "queue" && !edited) {
        extra.push(a.choice === c.gold_queue ? chip("act", "matches the label", "与标注一致")
                                             : chip("block", "label: " + c.gold_queue, "标注：" + CASES.queue_zh[c.gold_queue]));
      }
      out.appendChild(ansCard(id, a, extra));
    });
  }
  function runTri() {
    var c = CASES.triage.cases[TRI], state = triState();
    var edited = state.body !== c.state.body || state.subject !== (c.state.subject || "");
    $("tri-run").disabled = true; status();
    M.decide({ state: state, questions: CASES.triage.questions }).then(function (r) {
      $("tri-run").disabled = false;
      if (!r.ok) return fail(r);
      TRI_RES = { data: r.data, edited: edited };
      var tm = r.data.timing || {};
      $("tri-time").innerHTML = "";
      $("tri-time").appendChild(bi("5 answers from one read · " + ms(tm.total_ms || r.ms) + " · " + r.data.usage.input_tokens + " tokens",
                                   "一次读取得到 5 个回答 · " + ms(tm.total_ms || r.ms) + " · " + r.data.usage.input_tokens + " token"));
      renderTriOut();
    });
  }

  // ---- inbox --------------------------------------------------------------------------------
  function renderBoard() {
    var board = $("inbox-board"); board.innerHTML = "";
    var folders = Object.keys(CASES.folders), cols = {};
    folders.forEach(function (f) {
      var col = el("div", { "class": "zh-folder" }), h = el("h3");
      h.appendChild(bi(FOLDER_EN[f] || f, f)); h.appendChild(el("small", null, "0"));
      col.appendChild(h); cols[f] = col; board.appendChild(col);
    });
    var unsorted = el("div", { "class": "zh-folder" }), uh = el("h3");
    uh.appendChild(bi("Unsorted", "未分拣")); uh.appendChild(el("small", null, "0")); unsorted.appendChild(uh);
    var ok = 0, done = 0;
    CASES.email.cases.forEach(function (c, i) {
      var res = INBOX[i], target = res ? cols[res.answers.folder.choice] : unsorted;
      var b = el("button", { "class": "zh-item" + (res ? "" : " pending"), type: "button",
                             "aria-selected": MAIL_SEL === i ? "true" : "false" });
      b.appendChild(el("span", { "class": "s", lang: "zh-CN" }, c.state.subject));
      var ch = el("span", { "class": "zh-chips" });
      if (res) {
        done++;
        var pr = res.answers.priority, top = argmax(pr.probabilities);
        ch.appendChild(chip(top === "2" ? "block" : top === "1" ? "review" : "", "priority " + pr.legend[top], "优先级 " + pr.legend[top]));
        if (res.answers.reply_today.noul >= 0.5) ch.appendChild(chip("review", "reply today", "今天回复"));
        var good = res.answers.folder.choice === c.gold_folder; if (good) ok++;
        ch.appendChild(good ? chip("act", "✓", "✓") : chip("block", "expected " + (FOLDER_EN[c.gold_folder] || c.gold_folder), "应为 " + c.gold_folder));
      }
      b.appendChild(ch);
      b.addEventListener("click", function () { MAIL_SEL = i; renderBoard(); renderMail(); });
      target.appendChild(b);
      var n = target.querySelector("h3 small"); n.textContent = String(+n.textContent + 1);
    });
    if (done < CASES.email.cases.length) board.appendChild(unsorted);
    var sum = $("inbox-sum"); sum.innerHTML = "";
    if (done) {
      var tot = 0; Object.keys(INBOX).forEach(function (k) { tot += INBOX[k].timing.total_ms || 0; });
      sum.appendChild(bi(done + "/" + CASES.email.cases.length + " sorted · " + ok + " in the expected folder · avg " + ms(tot / done) + " per e-mail (3 questions each)",
                         "已分拣 " + done + "/" + CASES.email.cases.length + " · " + ok + " 封进了预期文件夹 · 每封平均 " + ms(tot / done) + "（每封 3 个问题）"));
    }
  }
  function renderMail() {
    var box = $("inbox-detail"), c = CASES.email.cases[MAIL_SEL], res = INBOX[MAIL_SEL];
    box.innerHTML = ""; box.hidden = MAIL_SEL == null;
    if (MAIL_SEL == null) return;
    box.appendChild(el("h3", { lang: "zh-CN" }, c.state.subject));
    box.appendChild(el("div", { "class": "body", lang: "zh-CN" }, c.state.body));
    if (!res) return;
    var grid = el("div", { "class": "zh-answers" });
    ["folder", "reply_today", "priority"].forEach(function (id) {
      var q = CASES.email.questions[id], card = el("div", { "class": "zh-ans" }), h = el("h3");
      h.appendChild(document.createTextNode(q.instructions)); card.appendChild(h);
      var a = res.answers[id], inner = el("div");
      if (a.type === "noul") M.bars(inner, { yes: a.noul, no: 1 - a.noul }, { labels: { yes: t("yes", "是"), no: t("no", "否") } });
      else if (a.type === "choice") M.bars(inner, a.probabilities, { order: Object.keys(q.criteria) });
      else M.bars(inner, a.probabilities, { labels: a.legend, order: Object.keys(a.legend) });
      card.appendChild(inner); grid.appendChild(card);
    });
    box.appendChild(grid);
  }
  function runInbox() {
    var btn = $("inbox-run"); btn.disabled = true; INBOX = {}; status(); renderBoard();
    var i = 0, n = CASES.email.cases.length;
    (function next() {
      if (i >= n) { btn.disabled = false; if (MAIL_SEL != null) renderMail(); return; }
      var c = CASES.email.cases[i];
      M.decide({ state: c.state, questions: CASES.email.questions }).then(function (r) {
        if (!r.ok) { btn.disabled = false; return fail(r); }
        INBOX[i] = r.data; i++; renderBoard(); next();
      });
    })();
  }

  // ---- intent -------------------------------------------------------------------------------
  function locName(loc) { var L = MASSIVE.locales[loc]; return L.native + " · " + t(L.en, L.zh); }
  function fillLangs() {
    var sel = $("int-lang"), cur = sel.value || "zh-CN"; sel.innerHTML = "";
    [[true, "In training", "训练中见过"], [false, "Not in training", "训练中没见过"]].forEach(function (g) {
      var og = el("optgroup", { label: t(g[1], g[2]) });
      Object.keys(MASSIVE.locales).filter(function (l) { return MASSIVE.locales[l].in_training === g[0]; }).forEach(function (l) {
        og.appendChild(el("option", { value: l }, locName(l) + " (" + l + ")"));
      });
      sel.appendChild(og);
    });
    sel.value = cur;
  }
  function fillItems() {
    var sel = $("int-item"), cur = sel.value || "0"; sel.innerHTML = "";
    MASSIVE.items.forEach(function (it, i) {
      sel.appendChild(el("option", { value: String(i) }, "[" + it.scenario + "] " + it.utt["en-US"]));
    });
    sel.value = cur;
  }
  function item() { return MASSIVE.items[+$("int-item").value || 0]; }
  function intentQuestion(it) {
    var crit = {}; it.candidates.forEach(function (k) { crit[k] = k.replace(/_/g, " "); });
    return { intent: { type: "choice", instructions: MASSIVE.instructions, criteria: crit } };
  }
  function showUtt() {
    var it = item(), loc = $("int-lang").value, u = $("int-utt");
    u.textContent = it.utt[loc];
    u.setAttribute("dir", MASSIVE.locales[loc].rtl ? "rtl" : "ltr");
    u.setAttribute("lang", loc);
    $("int-en").textContent = loc === "en-US" ? "" : "en: " + it.utt["en-US"];
    $("int-out").innerHTML = ""; $("int-time").textContent = "";
  }
  function renderIntent(data, it) {
    var out = $("int-out"); out.innerHTML = "";
    var a = data.answers.intent, row = el("div", { "class": "mj-row" });
    row.appendChild(el("b", null, a.choice));
    row.appendChild(a.choice === it.gold ? chip("act", "matches the label", "与标注一致") : chip("block", "label: " + it.gold, "标注：" + it.gold));
    out.appendChild(row);
    var bars = el("div"); bars.style.marginTop = "8px"; out.appendChild(bars);
    M.bars(bars, a.probabilities, { order: it.candidates });
  }
  function runIntent() {
    var it = item(), loc = $("int-lang").value;
    $("int-run").disabled = true; status();
    M.decide({ state: { utterance: it.utt[loc] }, questions: intentQuestion(it) }).then(function (r) {
      $("int-run").disabled = false;
      if (!r.ok) return fail(r);
      $("int-time").textContent = ms((r.data.timing || {}).total_ms || r.ms);
      renderIntent(r.data, it);
    });
  }
  function renderCells() {
    var box = $("int-cells"), it = item(); box.innerHTML = "";
    var keys = Object.keys(ALL), ok = 0, tr = [0, 0], un = [0, 0], tot = 0;
    Object.keys(MASSIVE.locales).forEach(function (loc) {
      var r = ALL[loc], L = MASSIVE.locales[loc];
      var b = el("button", { type: "button", "class": "zh-cell" + (r ? (r.ok ? " ok" : " bad") : ""), title: it.utt[loc] });
      var n = el("span", { "class": "n" }); n.appendChild(el("span", null, t(L.en, L.zh)));
      n.appendChild(el("small", null, L.in_training ? "•" : "")); b.appendChild(n);
      b.appendChild(el("span", { "class": "p" }, r ? r.choice + " " + M.pct(r.p) : "…"));
      b.addEventListener("click", function () { $("int-lang").value = loc; showUtt(); if (r) renderIntent(r.data, it); });
      box.appendChild(b);
      if (r) {
        tot += r.ms; if (r.ok) ok++;
        var g = L.in_training ? tr : un; g[1]++; if (r.ok) g[0]++;
      }
    });
    var s = $("int-allsum"); s.innerHTML = ""; s.hidden = !keys.length;
    if (keys.length) {
      s.appendChild(bi(ok + "/" + keys.length + " locales match the label · languages seen in training (•) " + tr[0] + "/" + tr[1] +
                       " · not seen " + un[0] + "/" + un[1] + " · avg " + ms(tot / keys.length),
                       ok + "/" + keys.length + " 个语种与标注一致 · 训练中见过的（•）" + tr[0] + "/" + tr[1] +
                       " · 没见过的 " + un[0] + "/" + un[1] + " · 平均 " + ms(tot / keys.length)));
    }
  }
  function runAll() {
    var btn = $("int-all");
    if (btn.getAttribute("data-running")) { STOP = true; return; }
    var it = item(), locs = Object.keys(MASSIVE.locales), i = 0;
    ALL = {}; STOP = false; status(); renderCells();
    btn.setAttribute("data-running", "1"); btn.innerHTML = ""; btn.appendChild(bi("Stop", "停止"));
    function done() {
      btn.removeAttribute("data-running"); btn.innerHTML = ""; btn.appendChild(bi("Ask in all 51 languages", "51 种语言都问一遍"));
    }
    (function next() {
      if (i >= locs.length || STOP) return done();
      var loc = locs[i];
      M.decide({ state: { utterance: it.utt[loc] }, questions: intentQuestion(it) }).then(function (r) {
        if (!r.ok) { done(); return fail(r); }
        var a = r.data.answers.intent;
        ALL[loc] = { choice: a.choice, p: a.probabilities[a.choice], ok: a.choice === it.gold, ms: (r.data.timing || {}).total_ms || r.ms, data: r.data };
        i++; renderCells(); next();
      });
    })();
  }

  // ---- wiring -------------------------------------------------------------------------------
  function tab(name) {
    [].forEach.call($("zh-tabs").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", b.getAttribute("data-tab") === name ? "true" : "false");
    });
    ["triage", "inbox", "intent"].forEach(function (p) { $("p-" + p).hidden = p !== name; });
    status();
    try { sessionStorage.setItem("jevstyle.zh.tab", name); } catch (e) {}
    if (name === "intent" && !MASSIVE) loadMassive();
  }
  function loadMassive() {
    fetch("/demo-data/chinese/massive_dev_subset.json").then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status); return r.json();
    }).then(function (d) {
      MASSIVE = d; fillLangs(); fillItems(); showUtt(); renderCells();
    }).catch(function (e) {
      status("MASSIVE data not found (" + e.message + "). Run scripts/demos/build_chinese.py.",
             "找不到 MASSIVE 数据（" + e.message + "）。请运行 scripts/demos/build_chinese.py。", true);
    });
  }
  function init() {
    fetch("cases.json").then(function (r) { return r.json(); }).then(function (c) {
      CASES = c; loadTri(); renderBoard();
    }).catch(function (e) { status("Could not load cases: " + e, "无法加载案例：" + e, true); });
    $("zh-tabs").addEventListener("click", function (e) { var b = e.target.closest("button"); if (b) tab(b.getAttribute("data-tab")); });
    $("tri-run").addEventListener("click", runTri);
    $("inbox-run").addEventListener("click", runInbox);
    $("int-run").addEventListener("click", runIntent);
    $("int-all").addEventListener("click", runAll);
    $("int-lang").addEventListener("change", showUtt);
    $("int-item").addEventListener("change", function () { ALL = {}; showUtt(); renderCells(); });
    $("zh-key-save").addEventListener("click", function () { M.setKey($("zh-key-input").value.trim()); $("zh-key").hidden = true; status(); });
    document.addEventListener("jevstyle:lang", function () {
      if (CASES) { renderTriList(); renderTriOut(); renderBoard(); if (MAIL_SEL != null) renderMail(); }
      if (MASSIVE) { fillLangs(); renderCells(); }
    });
    var saved = null; try { saved = sessionStorage.getItem("jevstyle.zh.tab"); } catch (e) {}
    if (saved && saved !== "triage" && $("p-" + saved)) tab(saved);
  }
  init();
})();
