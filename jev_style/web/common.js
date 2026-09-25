/* Jev-Style shared header, nav, EN/中文 toggle, theme toggle and small helpers. Owned by the Playground.
 *
 * Include on every page:  <link rel="stylesheet" href="/common.css"><script src="/common.js" defer></script>
 * Bilingual text:          <span lang="en">..</span><span lang="zh">..</span>   (English is the default)
 * Bilingual attributes:    data-placeholder-en / -zh, data-title-en / -zh, data-aria-label-en / -zh
 * Bilingual page title:    <title data-zh="中文标题">English title</title>
 * The nav lists Playground, Confidence and every demo from GET /api/demos.
 *
 * window.JevStyle:
 *   getLang() / setLang("en"|"zh"); document fires "jevstyle:lang" (detail = lang) on change
 *   t(en, zh)          -> string in the current language
 *   bi(en, zh)         -> <span> holding both languages (switches with the toggle, no re-render needed)
 *   getKey() / setKey(k)  API key kept in sessionStorage (only needed when the server runs with --api-key-env)
 *   decide(body)       -> Promise<{ok, status, data, ms}>  POST /v1/systemone with the key, never throws on HTTP errors
 *   confidence(probs)  -> (k * pmax - 1) / (k - 1), clipped to [0, 1]; probs = array or {option: p}
 *   bars(el, probs, opts) renders probability bars into el (opts.labels: {key: label}, opts.order: [keys])
 */
(function () {
  "use strict";
  var LANG_KEY = "jevstyle.lang", THEME_KEY = "jevstyle.theme", API_KEY = "jevstyle.apikey";

  function store(kind) { try { return window[kind]; } catch (e) { return null; } }
  function sget(kind, k) { try { var s = store(kind); return s ? s.getItem(k) : null; } catch (e) { return null; } }
  function sset(kind, k, v) {
    try { var s = store(kind); if (!s) return; if (v == null || v === "") s.removeItem(k); else s.setItem(k, v); } catch (e) {}
  }

  function getLang() { var l = sget("localStorage", LANG_KEY); return l === "zh" ? "zh" : "en"; }
  function applyLang(l) {
    var root = document.documentElement;
    root.setAttribute("data-lang", l);
    root.setAttribute("lang", l === "zh" ? "zh-CN" : "en");
    var attrs = ["placeholder", "title", "aria-label"];
    attrs.forEach(function (a) {
      var nodes = document.querySelectorAll("[data-" + a + "-en]");
      for (var i = 0; i < nodes.length; i++) {
        var v = nodes[i].getAttribute("data-" + a + "-" + l) || nodes[i].getAttribute("data-" + a + "-en");
        nodes[i].setAttribute(a, v);
      }
    });
    var title = document.querySelector("title");
    if (title) {
      if (!title.hasAttribute("data-en")) title.setAttribute("data-en", title.textContent);
      var tv = title.getAttribute("data-" + l);
      if (tv) document.title = tv;
    }
  }
  function setLang(l) {
    l = l === "zh" ? "zh" : "en";
    sset("localStorage", LANG_KEY, l);
    applyLang(l);
    document.dispatchEvent(new CustomEvent("jevstyle:lang", { detail: l }));
  }
  function t(en, zh) { return getLang() === "zh" && zh != null ? zh : en; }

  function getTheme() { var v = sget("localStorage", THEME_KEY); return v === "light" || v === "dark" ? v : "auto"; }
  function applyTheme(v) {
    if (v === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", v);
  }
  applyTheme(getTheme());
  applyLang(getLang());

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    for (var k in attrs || {}) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function bi(en, zh) {
    var s = el("span");
    s.appendChild(el("span", { lang: "en" }, en));
    s.appendChild(el("span", { lang: "zh" }, zh == null ? en : zh));
    return s;
  }

  function normPath(p) { return p.replace(/index\.html$/, ""); }
  function link(href, en, zh) {
    var a = el("a", { href: href });
    a.appendChild(el("span", { lang: "en" }, en));
    a.appendChild(el("span", { lang: "zh" }, zh));
    var here = normPath(location.pathname), target = normPath(href);
    if (here === target || (target !== "/" && target.indexOf("/demos/") === 0 &&
        here.indexOf(target.replace(/[^/]*$/, "")) === 0)) a.setAttribute("aria-current", "page");
    return a;
  }

  var THEME_LABEL = { auto: "◐", light: "☀", dark: "☾" };
  function build() {
    if (document.querySelector(".mj-header")) return;
    var header = el("header", { "class": "mj-header" });
    header.appendChild(el("a", { "class": "mj-brand", href: "/" }, "JevStyle"));
    var nav = el("nav", { "class": "mj-nav", "aria-label": "JevStyle" });
    nav.appendChild(link("/", "Playground", "试用"));
    nav.appendChild(link("/confidence.html", "Confidence", "置信度"));
    header.appendChild(nav);
    var tools = el("div", { "class": "mj-tools" });
    var theme = el("button", { "class": "mj-theme", type: "button", "data-title-en": "Theme: auto / light / dark",
      "data-title-zh": "主题：跟随系统 / 浅色 / 深色", "data-aria-label-en": "Theme", "data-aria-label-zh": "主题" },
      THEME_LABEL[getTheme()]);
    theme.addEventListener("click", function () {
      var order = ["auto", "light", "dark"], next = order[(order.indexOf(getTheme()) + 1) % 3];
      sset("localStorage", THEME_KEY, next === "auto" ? null : next);
      applyTheme(next);
      theme.textContent = THEME_LABEL[next];
    });
    var btn = el("button", { "class": "mj-lang", type: "button", "data-title-en": "Switch language",
      "data-title-zh": "切换语言", "data-aria-label-en": "Switch language", "data-aria-label-zh": "切换语言" });
    btn.appendChild(el("span", { lang: "en" }, "中文"));
    btn.appendChild(el("span", { lang: "zh" }, "EN"));
    btn.addEventListener("click", function () { setLang(getLang() === "en" ? "zh" : "en"); });
    tools.appendChild(theme);
    tools.appendChild(btn);
    header.appendChild(tools);
    document.body.insertBefore(header, document.body.firstChild);
    applyLang(getLang());
    engineNote(header);
    fetch("/api/demos").then(function (r) { return r.ok ? r.json() : { demos: [] }; }).then(function (d) {
      (d.demos || []).forEach(function (demo) { nav.appendChild(link(demo.url, demo.title_en, demo.title_zh)); });
      var cur = nav.querySelector("[aria-current]");
      if (cur && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = Math.max(0, cur.offsetLeft - nav.offsetLeft - 24);
    }).catch(function () {});
  }

  // One-line notice under the header whenever the answers do not come from a trained release.
  function engineNote(header) {
    fetch("/healthz").then(function (r) { return r.ok ? r.json() : null; }).then(function (h) {
      if (!h || document.querySelector(".mj-engine-note")) return;
      var note = null;
      if (h.model === "jev-style-fake") {
        note = bi("Fake engine: answers are pseudo-random numbers, not model output.",
                  "假引擎：答案是伪随机数，不是模型输出。");
      } else if (h.untrained) {
        note = bi("Untrained dev release: answers are meaningless until the trained Jev-Style v3 release is installed.",
                  "未训练的开发版：装上训练好的 Jev-Style v3 之前，答案没有意义。");
      }
      if (!note) return;
      var box = el("div", { "class": "mj-engine-note", role: "note" });
      box.appendChild(note);
      header.parentNode.insertBefore(box, header.nextSibling);
      applyLang(getLang());
    }).catch(function () {});
  }

  function getKey() { return sget("sessionStorage", API_KEY) || ""; }
  function setKey(k) { sset("sessionStorage", API_KEY, k); }

  function decide(body) {
    var headers = { "content-type": "application/json" }, key = getKey();
    if (key) headers.authorization = "Bearer " + key;
    var t0 = (window.performance || Date).now();
    return fetch("/v1/systemone", { method: "POST", headers: headers, body: JSON.stringify(body) })
      .then(function (r) {
        return r.text().then(function (txt) {
          var data;
          try { data = JSON.parse(txt); } catch (e) { data = { error: { code: "bad_response", message: txt.slice(0, 300) } }; }
          return { ok: r.ok, status: r.status, data: data, ms: (window.performance || Date).now() - t0 };
        });
      }, function (e) {
        return { ok: false, status: 0, data: { error: { code: "network_error", message: String(e) } }, ms: 0 };
      });
  }

  function values(probs) {
    if (Array.isArray(probs)) return probs.slice();
    return Object.keys(probs || {}).map(function (k) { return probs[k]; });
  }
  function confidence(probs) {
    var v = values(probs), k = v.length;
    if (k <= 1) return 1;
    var pmax = Math.max.apply(null, v);
    return Math.max(0, Math.min(1, (k * pmax - 1) / (k - 1)));
  }

  function pct(p) { return (p * 100 >= 99.95 || p * 100 < 0.05) && p !== 0 && p !== 1 ? (p * 100).toFixed(2) + "%" : (p * 100).toFixed(1) + "%"; }
  function bars(target, probs, opts) {
    opts = opts || {};
    var order = opts.order || Object.keys(probs), labels = opts.labels || {};
    var best = order.reduce(function (b, k) { return b == null || probs[k] > probs[b] ? k : b; }, null);
    var wrap = el("div", { "class": "mj-bars", role: "list" });
    order.forEach(function (k) {
      var p = +probs[k] || 0;
      var row = el("div", { "class": "row" + (k === best ? " top" : ""), role: "listitem" });
      var lbl = el("div", { "class": "lbl", title: typeof labels[k] === "string" ? labels[k] : k });
      if (labels[k] && typeof labels[k] === "object") lbl.appendChild(labels[k]); else lbl.textContent = labels[k] || k;
      var trk = el("div", { "class": "trk" }), fil = el("div", { "class": "fil" });
      fil.style.width = (Math.max(0, Math.min(1, p)) * 100).toFixed(2) + "%";
      trk.appendChild(fil);
      row.appendChild(lbl); row.appendChild(trk); row.appendChild(el("div", { "class": "val" }, pct(p)));
      wrap.appendChild(row);
    });
    target.innerHTML = "";
    target.appendChild(wrap);
    return wrap;
  }

  window.JevStyle = window.JevStyle || {};
  var M = window.JevStyle;
  M.getLang = getLang; M.setLang = setLang; M.t = t; M.bi = bi; M.el = el;
  M.getKey = getKey; M.setKey = setKey; M.decide = decide;
  M.confidence = confidence; M.bars = bars; M.pct = pct;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build); else build();
})();
