/* Snake demo: game rules, the model request and the greedy code bot. No DOM, no network.
 *
 * Browser: window.SnakeCore.   Node: require("./snake-core.js").
 * Python twin (same numbers, same text, checked by tests/demos/snake)
 *
 * Coordinates: x = column (0 = left), y = row (0 = top); "up" is y - 1.
 * Each tick code builds the state text and the options (the four moves minus the 180-degree
 * reversal, and minus deadly moves when the filter is on) and computes facts for every move
 * (deadly? food distance before/after, free cells reachable after the move). The model is asked ONE
 * choice question; what each option's description says depends on the wording:
 *   v1 (default): only whether the move gets closer to the food ("gets closer to the food (distance 5→4)",
 *                 "moves away ...", "eats the food"; "deadly: ..." when the filter is off). Free cells are
 *                 computed and shown in the page, never sent.
 *   v0 (first version): every fact, including the free-cell count.
 * Measured with the trained model (seeds 1-5, 10x10, filter on): v0 mean score 1.4 (all stalled),
 * v1 mean 17.8 (greedy bot 18.8). The model (or the greedy bot) picks the move.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SnakeCore = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var ORDER = ["up", "down", "left", "right"];
  var DIRS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
  var OPPOSITE = { up: "down", down: "up", left: "right", right: "left" };
  var GRIDS = [8, 10, 12, 16];
  var QUESTION_ID = "move";
  var WORDINGS = {
    v1: "Which move gets the snake closer to the food? Each option says what the move does.",
    v0: "Which move best reaches the food while keeping the snake alive? Each option lists facts computed by code: " +
        "\"deadly\" means the game ends on that move; food distance is the Manhattan distance from the head to the food; " +
        "free cells reachable counts the empty cells the head can still reach after the move (more is safer)."
  };
  var DEFAULT_WORDING = "v1";
  var INSTRUCTIONS = WORDINGS[DEFAULT_WORDING];

  // ---- seedable RNG: xorshift32 on a scrambled seed (integer arithmetic only, so Python matches bit for bit)
  function normSeed(seed) {
    var n = Number(seed);
    if (!isFinite(n)) n = 0;
    n = Math.floor(Math.abs(n)) % 4294967296;
    return n >>> 0;
  }
  function rngInit(seed) {
    var s = (normSeed(seed) ^ 0x9e3779b9) >>> 0;
    if (s === 0) s = 1;
    for (var i = 0; i < 4; i++) s = xorshift(s);
    return s;
  }
  function xorshift(x) {
    x = (x ^ (x << 13)) >>> 0;
    x = (x ^ (x >>> 17)) >>> 0;
    x = (x ^ (x << 5)) >>> 0;
    return x;
  }

  // ---- game
  function newGame(opts) {
    opts = opts || {};
    var size = opts.size || 10;
    if (GRIDS.indexOf(size) < 0) throw new Error("grid size must be one of " + GRIDS.join(", "));
    var cx = Math.floor(size / 2), cy = Math.floor(size / 2);
    var g = {
      size: size, seed: normSeed(opts.seed == null ? 1 : opts.seed), rng: 0,
      snake: [[cx, cy], [cx - 1, cy], [cx - 2, cy]], dir: "right", food: null,
      score: 0, steps: 0, sinceFood: 0, stallLimit: opts.stallLimit || size * size * 2,
      over: false, reason: null, crash: null
    };
    g.rng = rngInit(g.seed);
    placeFood(g);
    return g;
  }
  function cloneGame(g) {
    var c = {};
    for (var k in g) c[k] = g[k];
    c.snake = g.snake.map(function (p) { return [p[0], p[1]]; });
    c.food = g.food ? [g.food[0], g.food[1]] : null;
    c.crash = g.crash ? [g.crash[0], g.crash[1]] : null;
    return c;
  }
  function key(x, y) { return x + "," + y; }
  function occupied(cells) {
    var o = {};
    for (var i = 0; i < cells.length; i++) o[key(cells[i][0], cells[i][1])] = true;
    return o;
  }
  function placeFood(g) {
    var occ = occupied(g.snake), empty = [];
    for (var y = 0; y < g.size; y++) for (var x = 0; x < g.size; x++) if (!occ[key(x, y)]) empty.push([x, y]);
    if (!empty.length) { g.food = null; return false; }
    g.rng = xorshift(g.rng);
    g.food = empty[g.rng % empty.length];
    return true;
  }
  function inside(g, x, y) { return x >= 0 && y >= 0 && x < g.size && y < g.size; }
  function dist(a, b) { return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]); }

  // empty cells the head can reach from (hx, hy), the head cell itself not counted
  function reachable(g, blocked, hx, hy) {
    var seen = {}, stack = [[hx, hy]], n = 0;
    seen[key(hx, hy)] = true;
    while (stack.length) {
      var p = stack.pop();
      for (var i = 0; i < ORDER.length; i++) {
        var d = DIRS[ORDER[i]], x = p[0] + d[0], y = p[1] + d[1], k = key(x, y);
        if (!inside(g, x, y) || seen[k] || blocked[k]) continue;
        seen[k] = true;
        n++;
        stack.push([x, y]);
      }
    }
    return n;
  }

  function analyse(g, move) {
    var head = g.snake[0], d = DIRS[move], nx = head[0] + d[0], ny = head[1] + d[1];
    var f = { move: move, reverse: g.snake.length > 1 && move === OPPOSITE[g.dir], deadly: false, cause: null,
              eats: false, distBefore: null, distAfter: null, reachable: null, freeTotal: null, length: g.snake.length };
    if (!inside(g, nx, ny)) { f.deadly = true; f.cause = "wall"; return f; }
    f.eats = !!g.food && nx === g.food[0] && ny === g.food[1];
    var body = f.eats ? g.snake : g.snake.slice(0, -1);
    var occ = occupied(body);
    if (occ[key(nx, ny)]) { f.deadly = true; f.cause = "body"; return f; }
    f.length = body.length + 1;
    if (g.food) { f.distBefore = dist(head, g.food); f.distAfter = dist([nx, ny], g.food); }
    occ[key(nx, ny)] = true;
    f.freeTotal = g.size * g.size - f.length;
    f.reachable = reachable(g, occ, nx, ny);
    return f;
  }

  function deadlyText(f) {
    return f.cause === "wall" ? "deadly: leaves the board (wall)" : "deadly: hits the snake's own body";
  }
  // Every fact code computed (the v0 option text; the page shows it as the code facts).
  function factText(f) {
    if (f.deadly) return deadlyText(f);
    var parts = ["safe"];
    if (f.eats) parts.push("eats the food");
    else if (f.distAfter != null) parts.push("food distance " + f.distBefore + "→" + f.distAfter +
                                             (f.distAfter < f.distBefore ? " (closer)" : " (farther)"));
    var r = "free cells reachable after the move: " + f.reachable + " of " + f.freeTotal;
    if (f.reachable < f.length) r += " (fewer than the snake's length " + f.length + ")";
    parts.push(r);
    return parts.join("; ");
  }

  // What the model is told about one move.
  function optionText(f, wording) {
    wording = wording || DEFAULT_WORDING;
    if (!WORDINGS[wording]) throw new Error("unknown wording " + wording);
    if (wording === "v0") return factText(f);
    if (f.deadly) return deadlyText(f);
    if (f.eats) return "eats the food";
    if (f.distAfter == null) return "no food on the board";
    return (f.distAfter < f.distBefore ? "gets closer to the food" : "moves away from the food") +
      " (distance " + f.distBefore + "\u2192" + f.distAfter + ")";
  }

  function cellsText(cells) {
    return cells.map(function (p) { return "(" + p[0] + "," + p[1] + ")"; }).join(" ");
  }
  function boardText(g) {
    var rows = [], occ = {}, i;
    for (i = 1; i < g.snake.length; i++) occ[key(g.snake[i][0], g.snake[i][1])] = "o";
    occ[key(g.snake[0][0], g.snake[0][1])] = "H";
    if (g.food) occ[key(g.food[0], g.food[1])] = "*";
    for (var y = 0; y < g.size; y++) {
      var row = "";
      for (var x = 0; x < g.size; x++) row += occ[key(x, y)] || ".";
      rows.push(row);
    }
    return rows.join("\n");
  }
  function stateText(g) {
    var m = g.size - 1;
    return [
      "Snake game, " + g.size + "x" + g.size + " grid. Cell (x,y): x = column 0-" + m + " from the left, y = row 0-" + m +
        " from the top. up = y-1, down = y+1, left = x-1, right = x+1.",
      "Snake, head first: " + cellsText(g.snake),
      "Direction: " + g.dir,
      "Food: " + (g.food ? cellsText([g.food]) : "none"),
      "Board (H head, o body, * food, . empty):",
      boardText(g)
    ].join("\n");
  }

  /* One decision point. mode: "ask" (request for the model), "forced" (one safe move: code takes it,
     the model is not asked), "no_safe" (filter on and every move is deadly: the game ends). */
  function buildDecision(g, opts) {
    var filter = !opts || opts.filter !== false;
    var wording = (opts && opts.wording) || DEFAULT_WORDING;
    if (!WORDINGS[wording]) throw new Error("unknown wording " + wording);
    var facts = {}, candidates = [], safe = [], excluded = {};
    ORDER.forEach(function (m) {
      var f = analyse(g, m);
      facts[m] = f;
      if (f.reverse) { excluded[m] = "reverse"; return; }
      candidates.push(m);
      if (!f.deadly) safe.push(m);
    });
    var offered = candidates.slice(), mode = "ask";
    if (filter) {
      candidates.forEach(function (m) { if (facts[m].deadly) excluded[m] = "deadly"; });
      offered = safe.slice();
      mode = safe.length === 0 ? "no_safe" : safe.length === 1 ? "forced" : "ask";
    }
    var request = null;
    if (mode === "ask") {
      var criteria = {};
      offered.forEach(function (m) { criteria[m] = optionText(facts[m], wording); });
      var q = {};
      q[QUESTION_ID] = { type: "choice", instructions: WORDINGS[wording], criteria: criteria };
      request = { state: stateText(g), questions: q };
    }
    return { mode: mode, filter: filter, wording: wording, facts: facts, candidates: candidates, safe: safe,
             offered: offered, excluded: excluded, request: request };
  }

  // Greedy code bot: shortest food distance among safe moves; ties -> more reachable cells -> ORDER.
  function greedyChoice(dec) {
    if (!dec.safe.length) return dec.candidates[0];
    var best = null;
    dec.safe.forEach(function (m) {
      var f = dec.facts[m], d = f.eats ? 0 : (f.distAfter == null ? 1e9 : f.distAfter);
      if (!best || d < best.d || (d === best.d && f.reachable > best.r)) best = { m: m, d: d, r: f.reachable };
    });
    return best.m;
  }
  // The move code applies without asking anyone (forced / no_safe), else null.
  function codeMove(dec) {
    if (dec.mode === "forced") return dec.offered[0];
    if (dec.mode === "no_safe") return dec.candidates[0];
    return null;
  }

  function applyMove(g, move) {
    if (g.over) throw new Error("game is over");
    if (!DIRS[move]) throw new Error("unknown move " + move);
    if (g.snake.length > 1 && move === OPPOSITE[g.dir]) throw new Error("illegal move: reversal " + move);
    var head = g.snake[0], d = DIRS[move], nx = head[0] + d[0], ny = head[1] + d[1];
    g.steps += 1;
    g.dir = move;
    if (!inside(g, nx, ny)) { g.over = true; g.reason = "wall"; g.crash = [nx, ny]; return { ate: false, over: true }; }
    var eats = !!g.food && nx === g.food[0] && ny === g.food[1];
    var body = eats ? g.snake : g.snake.slice(0, -1);
    if (occupied(body)[key(nx, ny)]) { g.over = true; g.reason = "body"; g.crash = [nx, ny]; return { ate: false, over: true }; }
    g.snake = [[nx, ny]].concat(body);
    if (eats) {
      g.score += 1;
      g.sinceFood = 0;
      if (!placeFood(g)) { g.over = true; g.reason = "full"; }
    } else {
      g.sinceFood += 1;
      if (g.sinceFood >= g.stallLimit) { g.over = true; g.reason = "stalled"; }
    }
    return { ate: eats, over: g.over };
  }

  // Problems with a /v1/systemone response for this decision ([] = fine).
  function checkResponse(data, dec) {
    var p = [];
    if (!data || typeof data !== "object") return ["response is not an object"];
    if (typeof data.model !== "string") p.push("model missing");
    var a = data.answers && data.answers[QUESTION_ID];
    if (!a || typeof a !== "object") return p.concat(["answers." + QUESTION_ID + " missing"]);
    if (a.type !== "choice") p.push("answer type is not choice");
    if (dec.offered.indexOf(a.choice) < 0) p.push("choice " + JSON.stringify(a.choice) + " was not offered");
    var probs = a.probabilities || {}, keys = Object.keys(probs), sum = 0;
    if (keys.length !== dec.offered.length || dec.offered.some(function (m) { return !(m in probs); }))
      p.push("probabilities do not cover exactly the offered moves");
    keys.forEach(function (k) {
      var v = probs[k];
      if (typeof v !== "number" || !isFinite(v) || v < 0 || v > 1) p.push("probability of " + k + " out of range");
      else sum += v;
    });
    if (Math.abs(sum - 1) > 1e-4) p.push("probabilities do not sum to 1");
    if (typeof a.confidence !== "number" || a.confidence < 0 || a.confidence > 1) p.push("confidence out of range");
    return p;
  }

  return {
    ORDER: ORDER, DIRS: DIRS, OPPOSITE: OPPOSITE, GRIDS: GRIDS, QUESTION_ID: QUESTION_ID, INSTRUCTIONS: INSTRUCTIONS,
    WORDINGS: WORDINGS, DEFAULT_WORDING: DEFAULT_WORDING, optionText: optionText, deadlyText: deadlyText,
    normSeed: normSeed, rngInit: rngInit, xorshift: xorshift, newGame: newGame, cloneGame: cloneGame,
    analyse: analyse, factText: factText, stateText: stateText, boardText: boardText, buildDecision: buildDecision,
    greedyChoice: greedyChoice, codeMove: codeMove, applyMove: applyMove, checkResponse: checkResponse
  };
});
