#!/usr/bin/env python3
"""wasm-manager: per-tenant WebAssembly app manager for NAN on Tinfoil.

Replaces the runsc/microVM backends. Nested container runtimes need privileged
namespace/mount/exec operations that Tinfoil confines away; a Wasm runtime needs
none of them. Each tenant app is a `wasmtime serve` subprocess bound to a unique
host loopback port. Isolation is provided by three independent layers, all
unprivileged: the Wasm sandbox (memory-safe, no ambient authority), the OS
process boundary (separate PID, own process group), and curated WASI (no
filesystem, no host env, no network beyond the served HTTP socket).

It speaks the SAME HTTP contract the supervisor already uses for the "vm"
backend, so the supervisor needs no change:

  POST   /vms   {image, share, name?, appPort?}
                 -> 201 {id, status, endpoint, hostPort, sshHostPort, ...}
  DELETE /vms/:id        -> {id, deleted: true}
  GET    /vms/:id | /vms | /health | /capacity | /catalog | /debug/env

Notes:
- `image` is reinterpreted as a Wasm APP REFERENCE: a catalog id (resolved to a
  baked-in, attested .wasm under APPS_DIR) or an absolute path to a .wasm. No
  arbitrary runtime URL fetch: the catalog is baked into the (attested) image so
  what runs is exactly what was measured at deploy.
- `sshHostPort` is always 0. A Wasm app is not an OS; there is nothing to SSH
  into. The supervisor already tolerates sshPort 0.
- Apps must be wasi:http components (what `wasmtime serve` runs). A WASIX/wasmer
  socket-server launcher can be added behind the same LAUNCHER seam later.
"""
import http.server
import json
import os
import pathlib
import resource
import signal
import socket
import subprocess
import threading
import time
import uuid

# ---- config ---------------------------------------------------------------- #
PORT         = int(os.environ.get("WASM_MANAGER_PORT", "8091"))   # same port the supervisor expects
WASMTIME     = os.environ.get("WASMTIME_BIN", "wasmtime")
APPS_DIR     = pathlib.Path(os.environ.get("WASM_APPS_DIR", "/opt/nan/apps"))
CATALOG_PATH = pathlib.Path(os.environ.get("WASM_CATALOG", str(APPS_DIR / "catalog.json")))
HOST_IP      = os.environ.get("WASM_HOST_IP", "127.0.0.1")
PORT_LO      = int(os.environ.get("WASM_PORT_LO", "20000"))
PORT_HI      = int(os.environ.get("WASM_PORT_HI", "40000"))
NODE_VCPUS   = int(os.environ.get("NODE_VCPUS", "16"))
NODE_RAM_GB  = int(os.environ.get("NODE_RAM_GB", "64"))
DEF_MEM_MB   = int(os.environ.get("WASM_APP_MEM_MB", "512"))       # per-app RLIMIT_AS ceiling
READY_SECS   = float(os.environ.get("WASM_READY_TIMEOUT", "20"))
MOCK         = os.environ.get("WASM_MOCK", "") not in ("", "0", "false")
LOG_DIR      = pathlib.Path(os.environ.get("WASM_LOG_DIR", "/tmp/nan-wasm-logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_apps = {}    # id -> record


# ---- helpers --------------------------------------------------------------- #
def _load_catalog() -> dict:
    """Map of app-id -> {file, name, description, mem_mb?}. Baked-in + attested."""
    try:
        data = json.loads(CATALOG_PATH.read_text())
        return {a["id"]: a for a in data.get("apps", [])}
    except Exception:
        return {}


def _resolve_wasm(ref: str) -> pathlib.Path:
    """Resolve an app reference to a .wasm path. Catalog id first, then an
    absolute path INSIDE APPS_DIR. Never fetches from the network."""
    cat = _load_catalog()
    if ref in cat:
        p = (APPS_DIR / cat[ref]["file"]).resolve()
    else:
        p = pathlib.Path(ref).resolve()
    # containment: only allow paths under APPS_DIR
    if APPS_DIR.resolve() not in p.parents and p != APPS_DIR.resolve():
        raise ValueError(f"app '{ref}' is not in the catalog and not under {APPS_DIR}")
    if not p.is_file():
        raise ValueError(f"wasm module not found for app '{ref}' ({p})")
    return p


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST_IP, 0))
        return s.getsockname()[1]


def _port_open(port: int, timeout=0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((HOST_IP, port))
            return True
        except OSError:
            return False


def _used_share() -> float:
    return round(sum(r["share"] for r in _apps.values() if r["status"] in ("starting", "running")), 4)


def _capacity() -> dict:
    used = _used_share()
    return {"maxShare": 1.0, "usedShare": used,
            "vcpusFree": round(NODE_VCPUS * (1.0 - used), 2),
            "ramGbFree": round(NODE_RAM_GB * (1.0 - used), 2),
            "apps": len(_apps)}


def _rlimits(mem_mb: int):
    """preexec: cap address space (unprivileged, no cgroups) and put the app in
    its own session so teardown can kill the whole group cleanly."""
    def _apply():
        os.setsid()
        soft = mem_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
        except (ValueError, OSError):
            pass
    return _apply


# ---- lifecycle ------------------------------------------------------------- #
def launch(app_ref: str, name: str, share: float, mem_mb: int) -> dict:
    vid = "app_" + uuid.uuid4().hex[:9]
    port = _free_port()
    log_path = LOG_DIR / f"{vid}.log"
    rec = {"id": vid, "name": name or vid, "app": app_ref, "share": share,
           "hostPort": port, "sshHostPort": 0,
           "endpoint": f"http://{HOST_IP}:{port}", "status": "starting",
           "createdAt": time.time(), "_proc": None, "_log": str(log_path),
           "error": None, "mem_mb": mem_mb}
    with _lock:
        _apps[vid] = rec

    if MOCK:
        # Stand up a trivial responder so the full supervisor path is testable.
        rec["_proc"] = _mock_server(port, vid)
        rec["status"] = "running"
        return rec

    try:
        wasm = _resolve_wasm(app_ref)
    except ValueError as e:
        rec["status"], rec["error"] = "failed", str(e)
        return rec

    # `wasmtime serve` runs a wasi:http component and owns the HTTP listener.
    # Default grants are minimal: no --dir (no fs), no --env (no host env),
    # only the served socket. That is the sandbox; we add nothing.
    cmd = [WASMTIME, "serve", "-Scli", "-Shttp", "--addr", f"{HOST_IP}:{port}", str(wasm)]
    logf = open(log_path, "wb")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf,
                                stderr=logf, preexec_fn=_rlimits(mem_mb))
    except Exception as e:
        rec["status"], rec["error"] = "failed", f"spawn: {e}"
        logf.close()
        return rec
    rec["_proc"] = proc

    # readiness: port accepts, or the process dies first.
    deadline = time.time() + READY_SECS
    while time.time() < deadline:
        if proc.poll() is not None:
            rec["status"] = "failed"
            rec["error"] = _log_tail(log_path) or f"wasmtime exited {proc.returncode}"
            return rec
        if _port_open(port):
            rec["status"] = "running"
            return rec
        time.sleep(0.1)
    # timed out: keep it but flag; supervisor can decide
    rec["status"] = "running" if _port_open(port) else "failed"
    if rec["status"] == "failed":
        rec["error"] = "did not open port in time; " + (_log_tail(log_path) or "")
        _kill(rec)
    return rec


def _mock_server(port: int, vid: str):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"nan-wasm-ok {vid}".encode())
        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer((HOST_IP, port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _log_tail(path, n=800) -> str:
    try:
        return path.read_text(errors="replace")[-n:].strip()
    except Exception:
        return ""


def _kill(rec):
    p = rec.get("_proc")
    if p is None:
        return
    try:
        if MOCK:
            p.shutdown()
            return
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def teardown(vid: str) -> bool:
    with _lock:
        rec = _apps.pop(vid, None)
    if rec is None:
        return False
    _kill(rec)
    return True


def _public(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# ---- HTTP contract --------------------------------------------------------- #
class Handler(http.server.BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ok": True, "runtime": "wasmtime",
                                    "version": _wasmtime_version(), "mock": MOCK,
                                    "capacity": _capacity()})
        if self.path == "/capacity":
            return self._json(200, _capacity())
        if self.path == "/catalog":
            cat = _load_catalog()
            return self._json(200, {"apps": [
                {"id": a["id"], "name": a.get("name", a["id"]),
                 "description": a.get("description", "")} for a in cat.values()]})
        if self.path == "/debug/env":
            return self._json(200, _debug_env())
        if self.path == "/vms":
            with _lock:
                return self._json(200, {"vms": [_public(r) for r in _apps.values()]})
        if self.path.startswith("/vms/"):
            vid = self.path[len("/vms/"):]
            with _lock:
                rec = _apps.get(vid)
            return self._json(200, _public(rec)) if rec else self._json(404, {"error": "not found"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/vms":
            return self._json(404, {"error": "not found"})
        b = self._body()
        app_ref = b.get("image") or b.get("app")
        if not app_ref:
            return self._json(400, {"error": "missing app reference (image)"})
        try:
            share = float(b.get("share", 0.05))
        except (TypeError, ValueError):
            share = 0.05
        share = min(max(share, 0.0), 1.0)
        if _used_share() + share > 1.0 + 1e-6:
            return self._json(429, {"error": "insufficient capacity", "capacity": _capacity()})
        cat = _load_catalog()
        mem_mb = int(cat.get(app_ref, {}).get("mem_mb", DEF_MEM_MB))
        rec = launch(app_ref, b.get("name", ""), share, mem_mb)
        code = 201 if rec["status"] in ("starting", "running") else 500
        return self._json(code, _public(rec))

    def do_DELETE(self):
        if self.path.startswith("/vms/"):
            vid = self.path[len("/vms/"):]
            return self._json(200, {"id": vid, "deleted": teardown(vid)})
        return self._json(404, {"error": "not found"})


def _wasmtime_version() -> str:
    if MOCK:
        return "mock"
    try:
        r = subprocess.run([WASMTIME, "--version"], capture_output=True, text=True, timeout=10)
        return (r.stdout or r.stderr or "").strip().splitlines()[0] if (r.stdout or r.stderr) else ""
    except Exception as e:
        return f"err: {e}"


def _debug_env() -> dict:
    out = {"runtime": "wasmtime", "mock": MOCK, "apps_dir": str(APPS_DIR),
           "catalog": sorted(_load_catalog().keys()), "version": _wasmtime_version()}
    try:
        out["uname"] = " ".join(os.uname())
    except Exception as e:
        out["uname"] = f"err: {e}"
    # does `wasmtime serve` exist in this build?
    if not MOCK:
        try:
            r = subprocess.run([WASMTIME, "serve", "--help"], capture_output=True, text=True, timeout=10)
            out["serve_available"] = (r.returncode == 0)
        except Exception as e:
            out["serve_available"] = f"err: {e}"
    return out


def main():
    httpd = http.server.ThreadingHTTPServer((HOST_IP if HOST_IP else "0.0.0.0", PORT), Handler)
    print(f"wasm-manager on :{PORT} runtime=wasmtime mock={MOCK} apps_dir={APPS_DIR}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for vid in list(_apps):
            teardown(vid)


if __name__ == "__main__":
    main()
