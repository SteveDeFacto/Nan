#!/usr/bin/env python3
# ipfs-add-gateway.py — validating upload gateway in front of the local Kubo node.
#
# Why this exists: the browser's checks (extension, size, wasm preamble) are UX only
# — anyone can bypass them by POSTing straight to Kubo. So we do NOT expose Kubo's
# /api/v0/add to the internet. Caddy forwards ONLY /add-wasm to this gateway (on
# 127.0.0.1); the gateway validates the bytes, then adds+pins to Kubo with hardcoded
# params and returns {"cid": ...}. Kubo's API stays bound to localhost.
#
# Validation tiers:
#   Tier 1 (always): size cap + wasm magic (\0asm) + component *layer* field
#                    (0 = core module -> reject; 1 = component -> ok). Version-proof.
#   Tier 2 (if WASM_TOOLS is set to a `wasm-tools` binary): also run
#                    `wasm-tools validate` — authoritative structural validation.
#
# Pure stdlib, no pip deps. Runs on the VM next to Kubo.
#
# Env:
#   PORT            listen port (default 5051, bound to 127.0.0.1 only)
#   KUBO_API        Kubo API base (default http://127.0.0.1:5001)
#   MAX_WASM_BYTES  hard size cap (default 268435456 = 256 MiB)
#   ALLOW_ORIGIN    CORS origin for the browser (default https://nan.host)
#   WASM_TOOLS      path to a `wasm-tools` binary to enable Tier 2 (default: off)

import json, os, subprocess, tempfile, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT       = int(os.environ.get("PORT", "5051"))
KUBO_API   = os.environ.get("KUBO_API", "http://127.0.0.1:5001").rstrip("/")
MAX_BYTES  = int(os.environ.get("MAX_WASM_BYTES", str(256 * 1024 * 1024)))
ORIGIN     = os.environ.get("ALLOW_ORIGIN", "https://nan.host")
WASM_TOOLS = os.environ.get("WASM_TOOLS", "")


def preamble_error(b: bytes):
    """Tier 1: return an error string if the bytes aren't a wasm component, else None."""
    if len(b) < 8:
        return "too small to be a WebAssembly module"
    if b[0:4] != b"\x00asm":
        return "not a WebAssembly file (missing the \\0asm magic bytes)"
    layer = b[6] | (b[7] << 8)   # preamble after magic is version:u16 + layer:u16
    if layer == 0:
        return "this is a core wasm module, but NAN runs wasi:http components"
    if layer != 1:
        return "unrecognized wasm layer %d — expected a component" % layer
    return None


def wasm_tools_error(data: bytes):
    """Tier 2: authoritative validation via `wasm-tools validate` (if configured)."""
    if not WASM_TOOLS:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wasm") as tf:
        tf.write(data); tf.flush()
        r = subprocess.run([WASM_TOOLS, "validate", "--features", "all", tf.name],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return "wasm validation failed: " + (r.stderr or "").strip()[:200]
    return None


def kubo_add(data: bytes):
    """Add+pin to the local Kubo node with fixed params; return the CID."""
    boundary = "----nanwasm" + uuid.uuid4().hex
    pre = ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"app.wasm\"\r\n"
           "Content-Type: application/octet-stream\r\n\r\n" % boundary).encode()
    post = ("\r\n--%s--\r\n" % boundary).encode()
    req = urllib.request.Request(
        KUBO_API + "/api/v0/add?cid-version=1&pin=true",
        data=pre + data + post, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=180) as r:
        text = r.read().decode()
    last = json.loads(text.strip().splitlines()[-1])
    return last.get("Hash") or last.get("Cid") or last.get("cid")


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        return self._json(200 if self.path == "/healthz" else 404,
                          {"ok": True} if self.path == "/healthz" else {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/add-wasm":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json(411, {"error": "Content-Length required"})
        if length > MAX_BYTES:
            return self._json(413, {"error": "too large (max %d bytes)" % MAX_BYTES})
        data = self.rfile.read(length)
        if len(data) != length:
            return self._json(400, {"error": "short read"})

        err = preamble_error(data) or wasm_tools_error(data)
        if err:
            return self._json(415, {"error": err})
        try:
            cid = kubo_add(data)
        except Exception as e:  # noqa: BLE001 — surface any upstream failure to the client
            return self._json(502, {"error": "ipfs add failed: %s" % e})
        if not cid:
            return self._json(502, {"error": "ipfs returned no CID"})
        return self._json(200, {"cid": cid})

    def log_message(self, *a):  # quiet; systemd/journal captures stdout if needed
        pass


if __name__ == "__main__":
    print("wasm add-gateway on 127.0.0.1:%d -> %s%s" %
          (PORT, KUBO_API, " (wasm-tools on)" if WASM_TOOLS else " (header-only)"), flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
