/* Confidence Explorer: sliders over 3-5 option probabilities -> confidence = (k * pmax - 1) / (k - 1). */
(function () {
  "use strict";
  var M = window.JevStyle, el = M.el;
  var $ = function (id) { return document.getElementById(id); };
  var NAMES = ["A", "B", "C", "D", "E"];
  var S = { k: 3, p: [0.62, 0.25, 0.13] };
  var PRESETS = {
    sure: function (k) { return spread(0.94, k); },
    lean: function (k) { return spread(0.6, k); },
    split: function (k) { var r = [0.48, 0.46]; while (r.length < k) r.push(0.06 / (k - 2)); return r; },
    flat: function (k) { var r = []; for (var i = 0; i < k; i++) r.push(1 / k); return r; }
  };
  function spread(top, k) { var r = [top]; for (var i = 1; i < k; i++) r.push((1 - top) / (k - 1)); return r; }

  function conf(p) { return M.confidence(p); }
  function argmax(p) { var b = 0; for (var i = 1; i < p.length; i++) if (p[i] > p[b]) b = i; return b; }

  function setOne(i, v) {
    var p = S.p, rest = 1 - p[i], want = 1 - v;
    for (var j = 0; j < p.length; j++) {
      if (j === i) continue;
      p[j] = rest > 1e-9 ? p[j] * want / rest : want / (p.length - 1);
    }
    p[i] = v;
  }

  var rows = [];
  function renderSliders() {
    var box = $("sliders"); box.innerHTML = ""; rows = [];
    S.p.forEach(function (v, i) {
      var row = el("label", { "class": "cx-slider" });
      row.appendChild(el("span", { "class": "cx-name mono" }, NAMES[i]));
      var r = el("input", { type: "range", min: "0", max: "1000", step: "1", value: String(Math.round(v * 1000)),
        "aria-label": "p(" + NAMES[i] + ")" });
      var val = el("span", { "class": "cx-val mono" });
      r.addEventListener("input", function () { setOne(i, +r.value / 1000); update(i); });
      row.appendChild(r); row.appendChild(val);
      box.appendChild(row);
      rows.push({ range: r, val: val });
    });
  }

  function fmt(x, d) { return (+x).toFixed(d == null ? 2 : d); }
  var TIER = { act: ["act", "执行"], review: ["review", "复核"], block: ["block", "拦截"] };

  function update(skip) {
    var p = S.p, k = p.length, i0 = argmax(p), pm = p[i0], c = conf(p);
    rows.forEach(function (r, i) {
      if (i !== skip) r.range.value = String(Math.round(p[i] * 1000));
      r.val.textContent = (p[i] * 100).toFixed(1) + "%";
    });
    var ties = p.filter(function (x) { return Math.abs(x - pm) < 1e-6; }).length;
    $("winner").textContent = ties > 1 ? M.t("tie", "并列") : NAMES[i0];
    $("pmax").textContent = fmt(pm, 3);
    $("conf").textContent = fmt(c, 3);
    $("formula").textContent =
      "confidence = (k · p_max − 1) / (k − 1)\n" +
      "           = (" + k + " × " + fmt(pm, 3) + " − 1) / (" + k + " − 1)\n" +
      "           = " + fmt(k * pm - 1, 3) + " / " + (k - 1) + "\n" +
      "           = " + fmt(c, 3);
    var tier = c >= 0.8 ? "act" : c >= 0.4 ? "review" : "block";
    var t = $("tier"); t.innerHTML = "";
    var chip = el("span", { "class": "mj-chip cx-tier " + tier, "data-tier": tier }); chip.appendChild(M.bi(TIER[tier][0], TIER[tier][1]));
    t.appendChild(chip);
    drawPlot(k, pm, c);
  }

  var W = 320, H = 200, L = 34, R = 10, T = 10, B = 30;
  function X(p) { return L + p * (W - L - R); }
  function Y(c) { return H - B - c * (H - T - B); }
  function svg(tag, attrs) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var a in attrs) e.setAttribute(a, attrs[a]);
    return e;
  }
  function drawPlot(k, pm, c) {
    var s = $("plot"); while (s.firstChild) s.removeChild(s.firstChild);
    [0, 0.5, 1].forEach(function (v) {
      s.appendChild(svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), "class": "cx-grid" }));
      var tx = svg("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end", "class": "cx-tick" }); tx.textContent = v.toFixed(1); s.appendChild(tx);
    });
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      var tx = svg("text", { x: X(v), y: H - B + 16, "text-anchor": "middle", "class": "cx-tick" }); tx.textContent = v === 0 || v === 1 ? String(v) : v.toFixed(2); s.appendChild(tx);
    });
    var xl = svg("text", { x: (L + W - R) / 2, y: H - 2, "text-anchor": "middle", "class": "cx-tick" }); xl.textContent = "p_max"; s.appendChild(xl);
    for (var kk = 2; kk <= 5; kk++) {
      var x0 = 1 / kk;
      s.appendChild(svg("line", { x1: X(x0), y1: Y(0), x2: X(1), y2: Y(1), "class": "cx-line" + (kk === k ? " on" : "") }));
      if (kk !== k) {
        var lb = svg("text", { x: X(x0) - 2, y: Y(0) - 5 - (kk % 2 ? 0 : 10), "text-anchor": "middle", "class": "cx-klab" }); lb.textContent = "k=" + kk; s.appendChild(lb);
      }
    }
    var on = svg("text", { x: X(1 / k) - 2, y: Y(0) - 5 - (k % 2 ? 0 : 10), "text-anchor": "middle", "class": "cx-klab on" }); on.textContent = "k=" + k; s.appendChild(on);
    s.appendChild(svg("line", { x1: X(pm), x2: X(pm), y1: Y(0), y2: Y(c), "class": "cx-drop" }));
    s.appendChild(svg("circle", { cx: X(pm), cy: Y(c), r: 5.5, "class": "cx-dot" }));
  }

  function setK(k) {
    var old = S.p.slice().sort(function (a, b) { return b - a; });
    var p = PRESETS.flat(k);
    if (old.length) { p = spread(Math.max(1 / k, Math.min(1, old[0])), k); }
    S.k = k; S.p = p;
    paintK(); renderSliders(); update(-1);
  }
  function paintK() {
    var b = document.querySelectorAll("#k button");
    for (var i = 0; i < b.length; i++) b[i].setAttribute("aria-pressed", String(+b[i].getAttribute("data-k") === S.k));
  }

  function init() {
    Array.prototype.forEach.call(document.querySelectorAll("#k button"), function (b) {
      b.addEventListener("click", function () { setK(+b.getAttribute("data-k")); });
    });
    Array.prototype.forEach.call(document.querySelectorAll("#presets button"), function (b) {
      b.addEventListener("click", function () { S.p = PRESETS[b.getAttribute("data-p")](S.k); update(-1); });
    });
    paintK(); renderSliders(); update(-1);
    document.addEventListener("jevstyle:lang", function () { update(-1); });
    document.documentElement.setAttribute("data-cx-ready", "1");
  }
  window.JevStyleConfidence = { state: S, setOne: function (i, v) { setOne(i, v); update(-1); }, setK: setK };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
