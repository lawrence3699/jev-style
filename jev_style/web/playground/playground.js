/* Jev-Style Playground: edit a state and questions, run them on the local server, read the answers,
 * gate them into act / review / block with code shown on the page, share the request as a link.
 * Plain JS, no build step. Needs /common.js (window.JevStyle) and share.js (window.JevStyleShare). */
(function () {
  "use strict";
  var M = window.JevStyle, Share = window.JevStyleShare;
  var el = M.el, bi = M.bi;
  var $ = function (id) { return document.getElementById(id); };

  var S = {
    stateMode: "json", questions: {}, qMode: "form", preset: "", reference: null,
    presets: [], lastReq: null, lastRes: null, lastStatus: 0, lastRoundTrip: 0, errQuestion: null,
    gate: { mode: "conf", act: 0.8, review: 0.4, hi: 0.9, lo: 0.1 }, gateLang: "js", rawMode: "req",
    modelId: "jev-style-0.8b-decision-v3", auth: false
  };
  var TYPES = [["noul", "Yes / no", "是非"], ["choice", "Choice", "选择"], ["score", "Score", "打分"]];

  /* ---------- helpers ---------- */
  function seg(id, value, onPick) {
    var box = $(id), btns = box.querySelectorAll("button");
    function paint(v) { for (var i = 0; i < btns.length; i++) btns[i].setAttribute("aria-pressed", String(btns[i].getAttribute("data-mode") === v)); }
    for (var i = 0; i < btns.length; i++) btns[i].addEventListener("click", function (e) {
      var v = e.currentTarget.getAttribute("data-mode");
      if (onPick(v) !== false) paint(v);
    });
    paint(value);
    return paint;
  }
  function fmtMs(ms) { return ms == null ? "–" : (ms < 10 ? ms.toFixed(1) : Math.round(ms)) + " ms"; }
  function fmt(x, d) { return (+x).toFixed(d == null ? 2 : d); }
  function isObj(v) { return v && typeof v === "object" && !Array.isArray(v); }
  function descText(v) { return v == null ? "" : typeof v === "string" ? v : JSON.stringify(v); }
  function renameKey(obj, oldK, newK) {
    var out = {};
    Object.keys(obj).forEach(function (k) { out[k === oldK ? newK : k] = obj[k]; });
    return out;
  }
  function moveKey(obj, k, dir) {
    var keys = Object.keys(obj), i = keys.indexOf(k), j = i + dir;
    if (j < 0 || j >= keys.length) return obj;
    keys[i] = keys[j]; keys[j] = k;
    var out = {}; keys.forEach(function (x) { out[x] = obj[x]; });
    return out;
  }
  function uniqueId(base, obj) { var id = base, n = 2; while (obj[id]) id = base + "_" + n++; return id; }
  function markCustom() {
    if (S.preset) { S.preset = ""; S.reference = null; $("preset").value = ""; renderSource(); }
    scheduleRaw();
  }

  /* ---------- state editor ---------- */
  var paintStateMode;
  function setStateValue(v) {
    if (typeof v === "string") { S.stateMode = "text"; $("state").value = v; }
    else { S.stateMode = "json"; $("state").value = JSON.stringify(v, null, 2); }
    paintStateMode(S.stateMode);
    checkState();
  }
  function readState() {
    var txt = $("state").value;
    if (S.stateMode === "text") return { ok: true, value: txt };
    try {
      var v = JSON.parse(txt);
      if (v === null || (typeof v !== "object" && typeof v !== "string"))
        return { ok: false, msg: M.t("JSON state must be an object, an array or a string.", "JSON 状态必须是对象、数组或字符串。") };
      return { ok: true, value: v };
    } catch (e) { return { ok: false, msg: M.t("Invalid JSON: ", "JSON 无效：") + e.message }; }
  }
  function checkState() {
    var r = readState(), box = $("stateerr");
    box.hidden = r.ok; box.textContent = r.ok ? "" : r.msg;
    return r;
  }

  /* ---------- question form ---------- */
  function defaultCriteria(type, old) {
    if (type === "noul") return null;
    if (type === "choice") {
      if (Array.isArray(old)) { var o = {}; old.forEach(function (lv) { o[typeof lv === "string" ? lv : (lv.label || JSON.stringify(lv))] = isObj(lv) && lv.description ? lv.description : null; }); return o; }
      if (isObj(old) && ("true" in old || "false" in old)) return { yes: old["true"] || null, no: old["false"] || null };
      return { option_a: null, option_b: null };
    }
    if (isObj(old) && !("true" in old || "false" in old)) {
      var ks = Object.keys(old).slice(0, 10);
      if (ks.length >= 2) return ks.map(function (k) { return old[k] ? { label: k, description: descText(old[k]) } : k; });
    }
    return ["low", "medium", "high"];
  }
  function newQuestion(type) {
    var base = { noul: "is_it", choice: "which", score: "how_much" }[type];
    var q = { type: type, instructions: { noul: "Is this …?", choice: "Which … fits best?", score: "How … is it?" }[type] };
    if (type !== "noul") q.criteria = defaultCriteria(type);
    var qs = S.questions; qs[uniqueId(base, qs)] = q;
    renderForm(); markCustom();
  }

  function input(value, cls, ph) {
    var i = el("input", { "class": "mj-input " + (cls || ""), type: "text", spellcheck: "false" });
    i.value = value == null ? "" : value;
    if (ph) { i.setAttribute("data-placeholder-en", ph[0]); i.setAttribute("data-placeholder-zh", ph[1]); i.placeholder = M.t(ph[0], ph[1]); }
    return i;
  }
  function iconBtn(txt, en, zh, fn) {
    var b = el("button", { type: "button", "class": "mj-btn icon", "data-title-en": en, "data-title-zh": zh,
      "data-aria-label-en": en, "data-aria-label-zh": zh, title: M.t(en, zh), "aria-label": M.t(en, zh) }, txt);
    b.addEventListener("click", fn);
    return b;
  }

  function renderForm() {
    var box = $("qform"); box.innerHTML = "";
    var ids = Object.keys(S.questions);
    if (!ids.length) box.appendChild(el("p", { "class": "mj-small mj-muted" })).appendChild(bi("No questions yet.", "还没有问题。"));
    ids.forEach(function (id) { box.appendChild(questionCard(id)); });
  }

  function questionCard(id) {
    var q = S.questions[id];
    var card = el("div", { "class": "pg-q" + (S.errQuestion === id ? " bad" : ""), "data-qid": id });
    var head = el("div", { "class": "pg-qhead" });
    var idIn = input(id, "pg-qid", ["question id", "问题 id"]);
    idIn.setAttribute("aria-label", "id");
    idIn.addEventListener("change", function () {
      var nid = idIn.value.trim();
      if (!nid || (nid !== id && S.questions[nid])) { idIn.value = id; return; }
      S.questions = renameKey(S.questions, id, nid); renderForm(); markCustom();
    });
    head.appendChild(idIn);
    var tseg = el("div", { "class": "mj-seg", role: "group" });
    TYPES.forEach(function (t) {
      var b = el("button", { type: "button", "aria-pressed": String(q.type === t[0]) });
      b.appendChild(bi(t[1], t[2]));
      b.addEventListener("click", function () {
        if (q.type === t[0]) return;
        var old = q.criteria; q.type = t[0];
        var c = defaultCriteria(t[0], old);
        if (c == null) delete q.criteria; else q.criteria = c;
        renderForm(); markCustom();
      });
      tseg.appendChild(b);
    });
    head.appendChild(tseg);
    var tools = el("div", { "class": "pg-qtools" });
    tools.appendChild(iconBtn("↑", "Move up", "上移", function () { S.questions = moveKey(S.questions, id, -1); renderForm(); markCustom(); }));
    tools.appendChild(iconBtn("↓", "Move down", "下移", function () { S.questions = moveKey(S.questions, id, 1); renderForm(); markCustom(); }));
    tools.appendChild(iconBtn("×", "Remove question", "删除问题", function () { delete S.questions[id]; renderForm(); markCustom(); }));
    head.appendChild(tools);
    card.appendChild(head);

    var ins = el("textarea", { "class": "mj-textarea pg-ins", rows: "2", spellcheck: "false",
      "data-placeholder-en": "Instructions: the question to ask", "data-placeholder-zh": "指令：要问的问题" });
    ins.placeholder = M.t("Instructions: the question to ask", "指令：要问的问题");
    ins.value = typeof q.instructions === "string" ? q.instructions : JSON.stringify(q.instructions);
    ins.addEventListener("input", function () { q.instructions = ins.value; markCustom(); });
    card.appendChild(ins);

    var crit = el("div", { "class": "pg-crit" });
    if (q.type === "noul") renderNoul(q, crit);
    else if (q.type === "choice") renderChoice(q, crit);
    else renderScore(q, crit);
    card.appendChild(crit);
    return card;
  }

  function renderNoul(q, box) {
    var c = isObj(q.criteria) ? q.criteria : {};
    [["true", "Yes means (optional)", "“是”的含义（可选）"], ["false", "No means (optional)", "“否”的含义（可选）"]].forEach(function (k) {
      var row = el("div", { "class": "pg-crow" });
      row.appendChild(el("span", { "class": "pg-ctag" }, k[0] === "true" ? M.t("yes", "是") : M.t("no", "否")));
      var i = input(descText(c[k[0]]), "", [k[1], k[2]]);
      i.addEventListener("input", function () {
        var cur = isObj(q.criteria) ? q.criteria : {};
        if (i.value) cur[k[0]] = i.value; else delete cur[k[0]];
        if (Object.keys(cur).length) q.criteria = cur; else delete q.criteria;
        markCustom();
      });
      row.appendChild(i); box.appendChild(row);
    });
  }

  function renderChoice(q, box) {
    var keys = Object.keys(q.criteria || {});
    keys.forEach(function (name) {
      var row = el("div", { "class": "pg-crow" });
      var n = input(name, "pg-cname", ["option", "选项"]);
      n.addEventListener("change", function () {
        var nn = n.value.trim();
        if (!nn || (nn !== name && nn in q.criteria)) { n.value = name; return; }
        q.criteria = renameKey(q.criteria, name, nn); renderForm(); markCustom();
      });
      var d = input(descText(q.criteria[name]), "", ["description (optional)", "说明（可选）"]);
      d.addEventListener("input", function () { q.criteria[name] = d.value === "" ? null : d.value; markCustom(); });
      row.appendChild(n); row.appendChild(d);
      row.appendChild(iconBtn("×", "Remove option", "删除选项", function () {
        if (Object.keys(q.criteria).length <= 1) return;
        delete q.criteria[name]; renderForm(); markCustom();
      }));
      box.appendChild(row);
    });
    var add = el("button", { type: "button", "class": "mj-btn pg-addopt" });
    add.appendChild(bi("+ option", "+ 选项"));
    add.addEventListener("click", function () {
      if (keys.length >= 255) return;
      q.criteria[uniqueId("option_" + String.fromCharCode(97 + (keys.length % 26)), q.criteria)] = null;
      renderForm(); markCustom();
    });
    box.appendChild(add);
  }

  function renderScore(q, box) {
    var levels = Array.isArray(q.criteria) ? q.criteria : [];
    levels.forEach(function (lv, i) {
      var row = el("div", { "class": "pg-crow" });
      row.appendChild(el("span", { "class": "pg-ctag mono" }, String(i)));
      var isLD = isObj(lv) && typeof lv.label === "string";
      var label = typeof lv === "string" ? lv : isLD ? lv.label : JSON.stringify(lv);
      var desc = isLD && lv.description ? lv.description : "";
      var l = input(label, "pg-cname", ["level", "等级"]);
      var d = input(desc, "", ["description (optional)", "说明（可选）"]);
      function save() {
        q.criteria[i] = d.value ? { label: l.value, description: d.value } : l.value;
        markCustom();
      }
      l.addEventListener("input", save); d.addEventListener("input", save);
      row.appendChild(l); row.appendChild(d);
      row.appendChild(iconBtn("↑", "Move up", "上移", function () {
        if (i === 0) return; var t = q.criteria[i - 1]; q.criteria[i - 1] = q.criteria[i]; q.criteria[i] = t; renderForm(); markCustom();
      }));
      row.appendChild(iconBtn("×", "Remove level", "删除等级", function () {
        if (levels.length <= 2) return; q.criteria.splice(i, 1); renderForm(); markCustom();
      }));
      box.appendChild(row);
    });
    var add = el("button", { type: "button", "class": "mj-btn pg-addopt" });
    add.appendChild(bi("+ level", "+ 等级"));
    add.addEventListener("click", function () {
      if (levels.length >= 10) return;
      q.criteria.push("level " + levels.length); renderForm(); markCustom();
    });
    box.appendChild(add);
    box.appendChild(el("div", { "class": "mj-small mj-muted" })).appendChild(bi("Lowest first, 2–10 levels.", "从低到高，2–10 级。"));
  }

  /* ---------- questions JSON mode ---------- */
  var paintQMode;
  function setQMode(mode) {
    if (mode === S.qMode) return true;
    if (mode === "form") {
      var r = parseQJson();
      if (!r.ok) return false;
      S.questions = r.value;
    }
    S.qMode = mode;
    $("qform").hidden = mode !== "form";
    $("qadd").hidden = mode !== "form";
    $("qjsonwrap").hidden = mode !== "json";
    if (mode === "json") { $("qjson").value = JSON.stringify(S.questions, null, 2); $("qjsonerr").hidden = true; }
    else renderForm();
    return true;
  }
  function parseQJson() {
    var box = $("qjsonerr");
    try {
      var v = JSON.parse($("qjson").value);
      if (!isObj(v)) throw new Error(M.t("questions must be an object: {id: question}", "questions 必须是对象：{id: 问题}"));
      box.hidden = true;
      return { ok: true, value: v };
    } catch (e) {
      box.hidden = false; box.textContent = e.message;
      return { ok: false };
    }
  }
  function currentQuestions() {
    if (S.qMode === "json") { var r = parseQJson(); if (r.ok) S.questions = r.value; return r.ok ? S.questions : null; }
    return S.questions;
  }

  /* ---------- request ---------- */
  function buildRequest() {
    var st = checkState();
    if (!st.ok) return { ok: false };
    var qs = currentQuestions();
    if (!qs) return { ok: false };
    return { ok: true, body: { model: S.modelId, state: st.value, questions: qs } };
  }

  function run() {
    var r = buildRequest();
    if (!r.ok) return;
    var btn = $("run"); btn.disabled = true; btn.classList.add("busy");
    S.lastReq = r.body;
    return M.decide(r.body).then(function (res) {
      S.lastStatus = res.status; S.lastRes = res.data; S.lastRoundTrip = res.ms;
      btn.disabled = false; btn.classList.remove("busy");
      if (res.status === 401) showAuth(true);
      renderResult();
      S.rawMode = res.ok ? S.rawMode : "res";
      paintRaw(S.rawMode); renderRaw();
      var out = document.querySelector(".pg-out");
      if (out && window.matchMedia && window.matchMedia("(max-width: 900px)").matches && out.scrollIntoView)
        out.scrollIntoView({ block: "start" });
    });
  }

  /* ---------- results ---------- */
  function renderResult() {
    var data = S.lastRes || {}, err = data.error, ans = $("answers"), errBox = $("error");
    var prevBad = S.errQuestion;
    S.errQuestion = err && err.question ? err.question : null;
    if (prevBad !== S.errQuestion && S.qMode === "form") renderForm();
    if (err) {
      errBox.hidden = false; errBox.innerHTML = "";
      errBox.appendChild(el("div", { "class": "pg-errcode mono" }, (S.lastStatus || "") + " " + err.code + (err.question ? " · " + err.question : "")));
      errBox.appendChild(el("div", {}, err.message));
      if (err.code === "input_budget_exceeded") {
        var lim = Number(err.limit || 0).toLocaleString(), tok = Number(err.tokens || 0).toLocaleString();
        errBox.appendChild(el("div", { "class": "mj-small mj-muted" })).appendChild(err.kind === "head"
          ? bi("Question + options use " + tok + " tokens; the limit is " + lim + ". Nothing is truncated.",
               "问题加选项共 " + tok + " token，上限 " + lim + "。不会截断。")
          : bi("State + question use " + tok + " tokens; the limit is " + lim + ". Nothing is truncated.",
               "状态加问题共 " + tok + " token，上限 " + lim + "。不会截断。"));
      }
      if (err.code === "unauthorized") errBox.appendChild(el("div", { "class": "mj-small" })).appendChild(bi("Enter the server's API key above.", "请在上方填写服务的 API 密钥。"));
      ans.innerHTML = "";
      setMetrics(null);
      renderGateSummary();
      return;
    }
    errBox.hidden = true;
    setMetrics(data);
    ans.innerHTML = "";
    var answers = data.answers || {};
    Object.keys(answers).forEach(function (id) { ans.appendChild(answerCard(id, answers[id], (data.timing || {}).question_ms)); });
    applyGate();
  }

  function setMetrics(data) {
    var tm = data && data.timing || {};
    $("m-total").textContent = data ? fmtMs(tm.total_ms) : "–";
    $("m-prefix").textContent = data ? (tm.prefix_cached ? M.t("cached", "已缓存") :
      tm.prefix_ms ? fmtMs(tm.prefix_ms) : M.t("in one pass", "与问题一次完成")) : "–";
    $("m-tokens").textContent = data && data.usage ? Number(data.usage.input_tokens).toLocaleString() : "–";
    $("m-model").textContent = data ? (data.model || "") + " · " + (data.backend || "") : "–";
    $("m-total").title = data ? M.t("round trip in the browser: ", "浏览器往返：") + fmtMs(S.lastRoundTrip) : "";
  }

  function topOf(a) {
    if (a.type === "noul") return a.noul >= 0.5 ? "true" : "false";
    if (a.type === "choice") return a.choice;
    var p = a.probabilities || {}, best = null;
    Object.keys(p).forEach(function (k) { if (best == null || p[k] > p[best]) best = k; });
    return best;
  }
  function confOf(a) {
    if (a.type === "noul") return M.confidence([a.noul, 1 - a.noul]);
    return a.confidence != null ? a.confidence : M.confidence(a.probabilities);
  }

  function answerCard(id, a, qms) {
    var card = el("div", { "class": "mj-card pg-a", "data-qid": id });
    var head = el("div", { "class": "pg-ahead" });
    head.appendChild(el("span", { "class": "pg-aid mono" }, id));
    var tchip = el("span", { "class": "mj-chip" }); tchip.appendChild(bi.apply(null, (TYPES.filter(function (t) { return t[0] === a.type; })[0] || [0, a.type, a.type]).slice(1)));
    head.appendChild(tchip);
    head.appendChild(el("span", { "class": "mj-chip pg-tier", "data-tier": "" }));
    if (qms && qms[id] != null) head.appendChild(el("span", { "class": "pg-ams mj-small mj-muted mono" }, fmtMs(qms[id])));
    card.appendChild(head);

    var main = el("div", { "class": "pg-amain" });
    var q = (S.lastReq && S.lastReq.questions || {})[id] || {};
    var ref = S.reference && S.reference[id] ? S.reference[id] : null;
    var probs, order, labels = {};
    if (a.type === "noul") {
      probs = { "true": a.noul, "false": 1 - a.noul }; order = ["true", "false"];
      labels["true"] = bi("yes", "是"); labels["false"] = bi("no", "否");
      main.appendChild(el("span", { "class": "pg-big" })).appendChild(a.noul >= 0.5 ? bi("Yes", "是") : bi("No", "否"));
      main.appendChild(el("span", { "class": "pg-sub mono" }, "p(yes) = " + fmt(a.noul, 3)));
    } else if (a.type === "choice") {
      probs = a.probabilities; order = Object.keys(q.criteria || probs).filter(function (k) { return k in probs; });
      if (order.length !== Object.keys(probs).length) order = Object.keys(probs);
      main.appendChild(el("span", { "class": "pg-big" }, a.choice));
    } else {
      probs = a.probabilities; order = Object.keys(probs);
      order.forEach(function (k) { labels[k] = k + " · " + ((a.legend || {})[k] || ""); });
      var K = order.length;
      main.appendChild(el("span", { "class": "pg-big mono" }, fmt(a.score, 2)));
      main.appendChild(el("span", { "class": "pg-sub" }, "/ " + (K - 1) + "  " + ((a.legend || {})[topOf(a)] || "")));
    }
    var c = confOf(a);
    var cspan = el("span", { "class": "pg-conf mj-small" });
    cspan.appendChild(bi("confidence ", "置信度 ")); cspan.appendChild(el("b", { "class": "mono" }, fmt(c, 2)));
    main.appendChild(cspan);
    card.appendChild(main);
    if (a.type === "score") card.appendChild(scoreScale(a));
    var barsBox = el("div", { "class": "pg-abars" });
    M.bars(barsBox, probs, { order: order, labels: labels });
    card.appendChild(barsBox);
    if (ref) {
      var shown = ref.map(function (r) {
        if (a.type === "noul") return r === "true" ? M.t("yes", "是") : M.t("no", "否");
        if (a.type === "score") return r + " · " + ((a.legend || {})[r] || "");
        return r;
      }).join(", ");
      var hit = ref.indexOf(topOf(a)) >= 0;
      var rp = el("div", { "class": "pg-ref mj-small mj-muted" });
      rp.appendChild(bi("Dataset label: ", "数据集标注：")); rp.appendChild(el("b", {}, shown));
      rp.appendChild(el("span", { "class": "pg-refmark " + (hit ? "hit" : "miss") }, hit ? " ✓" : " ✗"));
      card.appendChild(rp);
    }
    card._answer = a;
    return card;
  }

  function scoreScale(a) {
    var K = Object.keys(a.probabilities).length;
    var wrap = el("div", { "class": "pg-scale", "aria-hidden": "true" });
    var track = el("div", { "class": "pg-strack" });
    for (var i = 0; i < K; i++) {
      var tick = el("span", { "class": "pg-stick" }); tick.style.left = (K > 1 ? i / (K - 1) * 100 : 0) + "%"; track.appendChild(tick);
    }
    var mk = el("span", { "class": "pg-smark" }); mk.style.left = (K > 1 ? a.score / (K - 1) * 100 : 0) + "%";
    track.appendChild(mk);
    wrap.appendChild(track);
    return wrap;
  }

  /* ---------- gate ---------- */
  var gateFn = null;
  function num(x) { return (+x).toFixed(2); }
  function gateCode(lang) {
    var g = S.gate, band = g.mode === "band";
    if (lang === "py") {
      return [
        band ? "YES_ACT, YES_BLOCK = " + num(g.hi) + ", " + num(g.lo) : null,
        "ACT, REVIEW = " + num(g.act) + ", " + num(g.review),
        "",
        "def confidence(a):",
        "    # (k * pmax - 1) / (k - 1) for k options",
        "    p = [a[\"noul\"], 1 - a[\"noul\"]] if a[\"type\"] == \"noul\" else list(a[\"probabilities\"].values())",
        "    k = len(p)",
        "    return max(0.0, (k * max(p) - 1) / (k - 1)) if k > 1 else 1.0",
        "",
        "def gate(a):",
        band ? "    if a[\"type\"] == \"noul\":" : null,
        band ? "        if a[\"noul\"] >= YES_ACT: return \"act\"" : null,
        band ? "        if a[\"noul\"] <= YES_BLOCK: return \"block\"" : null,
        band ? "        return \"review\"" : null,
        "    c = confidence(a)",
        "    if c >= ACT: return \"act\"",
        "    if c >= REVIEW: return \"review\"",
        "    return \"block\"",
        "",
        "tiers = {qid: gate(a) for qid, a in response[\"answers\"].items()}"
      ].filter(function (x) { return x !== null; }).join("\n");
    }
    return [
      band ? "const YES_ACT = " + num(g.hi) + ", YES_BLOCK = " + num(g.lo) + ";" : null,
      "const ACT = " + num(g.act) + ", REVIEW = " + num(g.review) + ";",
      "",
      "// (k * pmax - 1) / (k - 1) for k options",
      "function confidence(a) {",
      "  const p = a.type === \"noul\" ? [a.noul, 1 - a.noul] : Object.values(a.probabilities);",
      "  const k = p.length;",
      "  return k > 1 ? Math.max(0, (k * Math.max(...p) - 1) / (k - 1)) : 1;",
      "}",
      "",
      "function gate(a) {",
      band ? "  if (a.type === \"noul\") {" : null,
      band ? "    if (a.noul >= YES_ACT) return \"act\";" : null,
      band ? "    if (a.noul <= YES_BLOCK) return \"block\";" : null,
      band ? "    return \"review\";" : null,
      band ? "  }" : null,
      "  const c = confidence(a);",
      "  if (c >= ACT) return \"act\";",
      "  if (c >= REVIEW) return \"review\";",
      "  return \"block\";",
      "}"
    ].filter(function (x) { return x !== null; }).join("\n");
  }
  function compileGate() {
    var code = gateCode("js");
    try { gateFn = new Function(code + "\nreturn gate;")(); } catch (e) { gateFn = null; }
    $("gatecode").textContent = gateCode(S.gateLang);
  }
  var TIER = { act: ["act", "执行"], review: ["review", "复核"], block: ["block", "拦截"] };
  function applyGate() {
    var cards = document.querySelectorAll(".pg-a");
    for (var i = 0; i < cards.length; i++) {
      var chip = cards[i].querySelector(".pg-tier"), a = cards[i]._answer, tier;
      try { tier = gateFn ? gateFn(a) : "review"; } catch (e) { tier = "review"; }
      chip.className = "mj-chip pg-tier " + tier; chip.setAttribute("data-tier", tier);
      chip.innerHTML = ""; chip.appendChild(bi(TIER[tier][0], TIER[tier][1]));
    }
    renderGateSummary();
  }
  function renderGateSummary() {
    var box = $("gatesum"); box.innerHTML = "";
    var cards = document.querySelectorAll(".pg-a .pg-tier"), n = { act: 0, review: 0, block: 0 };
    for (var i = 0; i < cards.length; i++) n[cards[i].getAttribute("data-tier")]++;
    ["act", "review", "block"].forEach(function (t) {
      var c = el("span", { "class": "mj-chip " + t }); c.appendChild(bi(TIER[t][0], TIER[t][1]));
      c.appendChild(document.createTextNode(" " + n[t])); box.appendChild(c);
    });
    box.appendChild(el("span", { "class": "mj-small mj-muted" })).appendChild(
      bi("act = use the answer · review = ask a person · block = do not act",
         "执行 = 直接采用 · 复核 = 交给人工 · 拦截 = 不采取行动"));
  }
  function slider(key, en, zh, min, max) {
    var row = el("label", { "class": "pg-slider" });
    var name = el("span", { "class": "pg-sname" }); name.appendChild(bi(en, zh));
    var r = el("input", { type: "range", min: String(min), max: String(max), step: "0.01", value: String(S.gate[key]),
      "data-aria-label-en": en, "data-aria-label-zh": zh, "aria-label": M.t(en, zh) });
    var v = el("span", { "class": "pg-sval mono" }, num(S.gate[key]));
    r.addEventListener("input", function () {
      var x = +r.value, g = S.gate;
      g[key] = x;
      if (key === "act" && g.review > x) g.review = x;
      if (key === "review" && g.act < x) g.act = x;
      if (key === "hi" && g.lo > x) g.lo = x;
      if (key === "lo" && g.hi < x) g.hi = x;
      syncSliders(); compileGate(); applyGate();
    });
    row.appendChild(name); row.appendChild(r); row.appendChild(v);
    row._key = key; row._range = r; row._val = v;
    return row;
  }
  function renderSliders() {
    var box = $("gatesliders"); box.innerHTML = "";
    if (S.gate.mode === "band") {
      box.appendChild(slider("hi", "act if p(yes) ≥", "p(是) ≥ 时执行", 0.5, 1));
      box.appendChild(slider("lo", "block if p(yes) ≤", "p(是) ≤ 时拦截", 0, 0.5));
    }
    box.appendChild(slider("act", "act if confidence ≥", "置信度 ≥ 时执行", 0, 1));
    box.appendChild(slider("review", "review if confidence ≥", "置信度 ≥ 时复核", 0, 1));
  }
  function syncSliders() {
    var rows = $("gatesliders").children;
    for (var i = 0; i < rows.length; i++) { rows[i]._range.value = S.gate[rows[i]._key]; rows[i]._val.textContent = num(S.gate[rows[i]._key]); }
  }

  /* ---------- raw panel ---------- */
  var paintRaw, rawTimer = null;
  function scheduleRaw() { clearTimeout(rawTimer); rawTimer = setTimeout(function () { if (S.rawMode !== "res") renderRaw(); }, 150); }
  function pyRepr(v, ind) {
    ind = ind || "";
    var next = ind + "    ";
    if (v === null) return "None";
    if (v === true) return "True";
    if (v === false) return "False";
    if (typeof v === "number") return String(v);
    if (typeof v === "string") return JSON.stringify(v);
    if (Array.isArray(v)) return v.length ? "[\n" + v.map(function (x) { return next + pyRepr(x, next); }).join(",\n") + ",\n" + ind + "]" : "[]";
    var ks = Object.keys(v);
    return ks.length ? "{\n" + ks.map(function (k) { return next + JSON.stringify(k) + ": " + pyRepr(v[k], next); }).join(",\n") + ",\n" + ind + "}" : "{}";
  }
  function renderRaw() {
    var pre = $("raw"), note = $("rawnote"); note.textContent = "";
    var draft = buildRequest(), body = draft.ok ? draft.body : S.lastReq;
    if (S.rawMode === "res") {
      pre.textContent = S.lastRes ? JSON.stringify(S.lastRes, null, 2) : "";
      if (S.lastRes) note.textContent = "HTTP " + S.lastStatus;
      return;
    }
    if (!body) { pre.textContent = ""; return; }
    if (S.rawMode === "req") { pre.textContent = JSON.stringify(body, null, 2); return; }
    var origin = location.origin;
    if (S.rawMode === "curl") {
      var json = JSON.stringify(body).replace(/'/g, "'\\''");
      pre.textContent = "curl -s " + origin + "/v1/systemone \\\n  -H 'content-type: application/json' \\\n" +
        (S.auth ? "  -H \"authorization: Bearer $JEV_STYLE_API_KEY\" \\\n" : "") + "  -d '" + json + "'";
      return;
    }
    pre.textContent = "from jev_style import JevStyle\n\njs = JevStyle(base_url=" + JSON.stringify(origin) + ")" +
      (S.auth ? "  # reads JEV_STYLE_API_KEY" : "") + "\nstate = " + pyRepr(body.state) + "\nquestions = " + pyRepr(body.questions) +
      "\nout = js.decide(state, questions)\nfor qid, a in out[\"answers\"].items():\n    print(qid, a)";
  }

  /* ---------- presets, share, auth ---------- */
  function renderPresetOptions() {
    var sel = $("preset"), cur = S.preset; sel.innerHTML = "";
    sel.appendChild(el("option", { value: "" }, M.t("Custom", "自定义")));
    S.presets.forEach(function (p) { sel.appendChild(el("option", { value: p.slug }, M.t(p.title_en, p.title_zh))); });
    sel.value = cur;
  }
  function loadRequest(req, preset) {
    S.preset = preset ? preset.slug : ""; S.reference = preset ? preset.reference : null; S.errQuestion = null;
    setStateValue(req.state == null ? "" : req.state);
    S.questions = isObj(req.questions) ? JSON.parse(JSON.stringify(req.questions)) : {};
    if (S.qMode === "json") $("qjson").value = JSON.stringify(S.questions, null, 2); else renderForm();
    $("preset").value = S.preset;
    renderSource(); renderRaw();
  }
  function loadPreset(slug) {
    var p = S.presets.filter(function (x) { return x.slug === slug; })[0];
    if (p) loadRequest(p.request, p);
  }
  function renderSource() {
    var box = $("presetsrc"); box.innerHTML = "";
    var p = S.presets.filter(function (x) { return x.slug === S.preset; })[0];
    if (!p) return;
    var s = p.source;
    box.appendChild(bi("Example from ", "示例来自 "));
    box.appendChild(document.createTextNode(s.dataset + " · " + s.split + " · " + s.licence + " · "));
    var d = el("details", { "class": "pg-ids" });
    var sm = el("summary"); sm.appendChild(bi("row ids", "行 id")); d.appendChild(sm);
    d.appendChild(el("code", {}, Object.keys(s.row_ids).map(function (k) { return s.row_ids[k]; }).join("\n")));
    box.appendChild(d);
  }
  function gateHash() { var g = S.gate; return [g.mode, num(g.act), num(g.review), num(g.hi), num(g.lo)].join(","); }
  function readGateHash(txt) {
    var p = (txt || "").split(",");
    if (p.length !== 5 || (p[0] !== "conf" && p[0] !== "band")) return;
    var v = p.slice(1).map(Number);
    if (v.some(function (x) { return !(x >= 0 && x <= 1); })) return;
    S.gate = { mode: p[0], act: v[0], review: Math.min(v[1], v[0]), hi: v[2], lo: Math.min(v[3], v[2]) };
  }
  function share() {
    var r = buildRequest();
    if (!r.ok) return;
    var req = { state: r.body.state, questions: r.body.questions };
    return Share.encode(req).then(function (code) {
      var url = location.origin + location.pathname + "#r=" + code + "&g=" + gateHash();
      try { history.replaceState(null, "", url); } catch (e) {}
      $("sharebox").hidden = false;
      var inp = $("shareurl"); inp.value = url;
      var msg = $("sharemsg");
      msg.textContent = url.length.toLocaleString() + M.t(" characters", " 字符");
      function copied() { msg.textContent = M.t("Copied · ", "已复制 · ") + url.length.toLocaleString() + M.t(" characters", " 字符"); }
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(url).then(copied, function () { inp.select(); });
        else inp.select();
      } catch (e) { inp.select(); }
      return url;
    });
  }
  function showAuth(on) {
    S.auth = S.auth || on;
    var k = $("apikey"); k.hidden = !S.auth;
    if (S.auth) k.value = M.getKey();
  }

  /* ---------- wiring ---------- */
  function init() {
    paintStateMode = seg("statemode", S.stateMode, function (mode) {
      if (mode === S.stateMode) return;
      var ta = $("state"), txt = ta.value;
      if (mode === "json") {
        var t = txt.trim();
        var looksJson = t && (t[0] === "{" || t[0] === "[");
        if (!looksJson) ta.value = JSON.stringify({ text: txt }, null, 2);
      } else {
        try { var v = JSON.parse(txt); ta.value = typeof v === "string" ? v : JSON.stringify(v, null, 2); } catch (e) {}
      }
      S.stateMode = mode; checkState(); markCustom();
    });
    $("state").addEventListener("input", function () { checkState(); markCustom(); });
    paintQMode = seg("qmode", S.qMode, function (mode) { return setQMode(mode); });
    $("qjson").addEventListener("input", function () { if (parseQJson().ok) { S.questions = JSON.parse($("qjson").value); } markCustom(); });
    Array.prototype.forEach.call(document.querySelectorAll("#qadd [data-add]"), function (b) {
      b.addEventListener("click", function () { newQuestion(b.getAttribute("data-add")); });
    });
    seg("gatemode", S.gate.mode, function (m) { S.gate.mode = m; renderSliders(); compileGate(); applyGate(); });
    seg("gatelang", S.gateLang, function (m) { S.gateLang = m; compileGate(); });
    paintRaw = seg("rawmode", S.rawMode, function (m) { S.rawMode = m; renderRaw(); });
    $("copyraw").addEventListener("click", function () {
      var txt = $("raw").textContent;
      try { navigator.clipboard.writeText(txt).then(function () { $("rawnote").textContent = M.t("Copied", "已复制"); }, function () {}); } catch (e) {}
    });
    $("run").addEventListener("click", run);
    $("share").addEventListener("click", share);
    $("preset").addEventListener("change", function () { if ($("preset").value) loadPreset($("preset").value); else markCustom(); });
    $("apikey").addEventListener("input", function () { M.setKey($("apikey").value.trim()); });
    document.addEventListener("keydown", function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run(); }
    });
    document.addEventListener("jevstyle:lang", function () {
      renderPresetOptions();
      if (S.qMode === "form") renderForm();
      checkState();
      if (S.lastRes) { renderResult(); }
      var sm = $("sharemsg"); if (sm) sm.textContent = "";
    });

    var hash = Share.parseHash(location.hash);
    readGateHash(hash.g);
    renderSliders(); compileGate(); renderGateSummary();
    var gm = document.querySelectorAll("#gatemode button");
    for (var i = 0; i < gm.length; i++) gm[i].setAttribute("aria-pressed", String(gm[i].getAttribute("data-mode") === S.gate.mode));

    fetch("/healthz").then(function (r) { return r.json(); }).then(function (h) {
      if (h.auth) showAuth(true);
      if (!S.auth || M.getKey()) return fetch("/v1/models", { headers: M.getKey() ? { authorization: "Bearer " + M.getKey() } : {} })
        .then(function (r) { return r.ok ? r.json() : null; }).then(function (m) {
          var d = m && m.data && m.data[0];
          if (d) { S.modelId = d.id; $("m-model").textContent = d.id + " · " + d.backend; }
        });
    }).catch(function () {});

    var presetsReady = fetch("/playground/presets.json").then(function (r) { return r.json(); })
      .then(function (d) { S.presets = d.presets || []; renderPresetOptions(); })
      .catch(function () { S.presets = []; renderPresetOptions(); });

    presetsReady.then(function () {
      if (hash.r) {
        return Share.decode(hash.r).then(function (req) { loadRequest(req, null); }, function (e) {
          var box = $("error"); box.hidden = false; box.textContent = M.t("Could not read the shared link: ", "无法读取分享链接：") + e.message;
          if (S.presets[0]) loadPreset(S.presets[0].slug);
        });
      }
      if (S.presets[0]) loadPreset(S.presets[0].slug);
      else loadRequest({ state: "", questions: {} }, null);
    }).then(function () { document.documentElement.setAttribute("data-pg-ready", "1"); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
