/* Share-link encoding for the Playground (no external library).
 *
 * A link is  /#r=<code>[&g=<gate>]  where <code> is one format letter followed by base64url text:
 *   "z" + base64url(deflate-raw(UTF-8 JSON))   when the browser has CompressionStream (all current browsers)
 *   "j" + base64url(UTF-8 JSON)                fallback, longer but needs nothing
 * The JSON is the systemone request {state, questions}. Python equivalent (tests use it):
 *   zlib.compressobj(9, zlib.DEFLATED, -15) / zlib.decompress(raw, -15), base64.urlsafe_b64encode without "=".
 */
(function () {
  "use strict";
  function b64url(bytes) {
    var s = "", chunk = 0x8000;
    for (var i = 0; i < bytes.length; i += chunk) s += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }
  function unb64url(text) {
    var s = text.replace(/-/g, "+").replace(/_/g, "/");
    while (s.length % 4) s += "=";
    var bin = atob(s), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function pipe(bytes, stream) {
    var body = new Blob([bytes]).stream().pipeThrough(stream);
    return new Response(body).arrayBuffer().then(function (b) { return new Uint8Array(b); });
  }
  var canZip = typeof CompressionStream === "function" && typeof DecompressionStream === "function";

  function encode(obj) {
    var bytes = new TextEncoder().encode(JSON.stringify(obj));
    if (!canZip) return Promise.resolve("j" + b64url(bytes));
    return pipe(bytes, new CompressionStream("deflate-raw")).then(function (z) {
      return z.length < bytes.length ? "z" + b64url(z) : "j" + b64url(bytes);
    }, function () { return "j" + b64url(bytes); });
  }
  function decode(code) {
    try {
      var kind = code.charAt(0), bytes = unb64url(code.slice(1));
      if (kind === "j") return Promise.resolve(JSON.parse(new TextDecoder().decode(bytes)));
      if (kind === "z") {
        if (!canZip) return Promise.reject(new Error("this browser cannot read compressed links"));
        return pipe(bytes, new DecompressionStream("deflate-raw")).then(function (raw) {
          return JSON.parse(new TextDecoder().decode(raw));
        });
      }
      return Promise.reject(new Error("unknown link format"));
    } catch (e) { return Promise.reject(e); }
  }
  function parseHash(hash) {
    var out = {};
    (hash || "").replace(/^#/, "").split("&").forEach(function (part) {
      var i = part.indexOf("=");
      if (i > 0) out[part.slice(0, i)] = part.slice(i + 1);
    });
    return out;
  }
  window.JevStyleShare = { encode: encode, decode: decode, parseHash: parseHash, compressed: canZip };
})();
