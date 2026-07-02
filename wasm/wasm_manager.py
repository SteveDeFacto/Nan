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
- `image` is reinterpreted as a Wasm APP REFERENCE:
    * a catalog id (baked-in, attested .wasm under APPS_DIR), or
    * `ipfs://<cid>` — fetched from IPFS_GATEWAY and VERIFIED: we pull the DAG as a
      CAR, check every block hashes to its CID, and reassemble the file rooted at
      the requested CID (see ipfs_fetch.py). A tampering gateway fails the hash
      check, so "what ran == this exact CID" holds without trusting the gateway.
      The verified CID is what the supervisor folds into attestation.
    * an absolute path to a .wasm already under APPS_DIR.
- `sshHostPort` is always 0. A Wasm app is not an OS; there is nothing to SSH
  into. The supervisor already tolerates sshPort 0.
- Apps must be wasi:http components (what `wasmtime serve` runs). A WASIX/wasmer
  socket-server launcher can be added behind the same LAUNCHER seam later.
"""
import http.server
import json
import os
import pathlib
import re
import resource
import signal
import socket
import subprocess
import threading
import time
import uuid

try:
    import ipfs_fetch   # local module (same dir): fetch + verify a wasm by IPFS CID
except Exception as _e:   # optional feature — never let a missing module take down the manager
    ipfs_fetch = None
    print(f"[wasm-manager] run-by-CID disabled: {_e}", flush=True)

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
DEF_MEM_MB   = int(os.environ.get("WASM_APP_MEM_MB", "512"))       # per-app guest linear-memory ceiling (wasmtime -W max-memory-size)
READY_SECS   = float(os.environ.get("WASM_READY_TIMEOUT", "20"))
MOCK         = os.environ.get("WASM_MOCK", "") not in ("", "0", "false")
LOG_DIR      = pathlib.Path(os.environ.get("WASM_LOG_DIR", "/tmp/nan-wasm-logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
# run-by-CID: fetch an app's bytes from IPFS and verify they hash to the CID.
IPFS_GATEWAY   = os.environ.get("IPFS_GATEWAY", "https://ipfs.nan.host").rstrip("/")
WASM_MAX_BYTES = int(os.environ.get("WASM_MAX_BYTES", str(256 * 1024 * 1024)))  # cap on a fetched app
IPFS_TIMEOUT   = float(os.environ.get("IPFS_FETCH_TIMEOUT", "120"))
# firewall: per-version ports config from the catalog. Declarable range keeps tenant
# apps off the supervisor (8080), this manager (8091), and the manager-assigned
# serve range (PORT_LO..PORT_HI), so a declared port can never shadow infrastructure
# or another tenant's assigned HTTP socket.
PORT_MIN_DECL  = 1024
PORT_MAX_DECL  = 19999
RESERVED_PORTS = {8080, 8091}
AUDIT_SECS     = float(os.environ.get("WASM_AUDIT_INTERVAL", "10"))

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


def _check_component(data: bytes):
    """Reject anything that isn't a wasi:http *component* before we try to run it
    (same preamble check as the upload gateway; gives a clear error vs a wasmtime
    crash). Layer field: 0 = core module, 1 = component."""
    if len(data) < 8 or data[0:4] != b"\x00asm":
        raise ValueError("fetched bytes are not a WebAssembly file")
    layer = data[6] | (data[7] << 8)
    if layer == 0:
        raise ValueError("fetched a core wasm module, not a wasi:http component")
    if layer != 1:
        raise ValueError(f"unrecognized wasm layer {layer} (expected a component)")


def _resolve_cid(cid: str) -> pathlib.Path:
    """Fetch `cid` from IPFS, verify the bytes hash to it, cache under APPS_DIR, run."""
    safe = re.sub(r"[^A-Za-z0-9]", "", cid)
    if not safe:
        raise ValueError(f"bad ipfs cid '{cid}'")
    p = (APPS_DIR / f"ipfs-{safe}.wasm").resolve()
    if p.is_file():
        return p                                   # content-addressed cache hit
    if ipfs_fetch is None:
        raise ValueError("run-by-CID not available in this build (ipfs_fetch missing)")
    try:
        data = ipfs_fetch.fetch_verified(cid, IPFS_GATEWAY, WASM_MAX_BYTES, IPFS_TIMEOUT)
    except ValueError:
        raise                                      # verification / size errors already have clear messages
    except Exception as e:                          # network / gateway errors -> ValueError so launch() reports it
        raise ValueError(f"ipfs fetch failed for {cid}: {e}")
    _check_component(data)
    APPS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)                                  # atomic publish into the cache
    return p


def _resolve_wasm(ref: str) -> pathlib.Path:
    """Resolve an app reference to a .wasm path: a catalog id (baked-in, attested),
    `ipfs://<cid>` (fetched + verified against the CID), or a path INSIDE APPS_DIR."""
    if ref.startswith("ipfs://"):
        cid = ref[len("ipfs://"):].split("/", 1)[0].split("?", 1)[0].strip()
        return _resolve_cid(cid)
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


def _preexec():
    """preexec: put the app in its own session so teardown can kill the whole
    group cleanly, and cap open files.

    We deliberately do NOT cap RLIMIT_AS. `wasmtime` reserves an enormous
    *virtual* address space (multi-TiB PROT_NONE guard/pooling regions) for fast
    linear-memory bounds-checking while touching almost no physical RAM, and on a
    many-core host it also reserves a worker-thread stack per CPU. Any RLIMIT_AS
    small enough to bound real memory instead makes those reservations fail,
    killing the runtime at startup (the "memory allocation of N bytes failed"
    abort). The guest's real memory is bounded on its linear memory via
    `wasmtime -W max-memory-size` in launch(); that is the only memory a tenant
    can grow, so it is the meaningful per-app cap."""
    os.setsid()
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
    except (ValueError, OSError):
        pass


# ---- lifecycle ------------------------------------------------------------- #
def _parse_ports(entries):
    """Parse a firewall config (list of 'http' | 'http:N' | 'tcp:N' | 'udp:N').

    Empty / just 'http' -> classic serve mode: `wasmtime serve` on a manager-
    assigned port, no wasi:sockets, the sandbox we've always had.
    Anything else -> run mode: the app is a long-running command component that
    binds its DECLARED ports itself via wasi:sockets ('http:N' = it serves HTTP
    on N and the supervisor proxies /x/:id there)."""
    http_port, tcp, udp, norm = None, set(), set(), []
    for e in entries or []:
        s = str(e).strip().lower()
        if not s or s == "http":
            continue
        m = re.fullmatch(r"(http|tcp|udp):(\d{1,5})", s)
        if not m:
            raise ValueError(f"bad port spec '{e}' (use http[:N] | tcp:N | udp:N)")
        n = int(m.group(2))
        if not (PORT_MIN_DECL <= n <= PORT_MAX_DECL) or n in RESERVED_PORTS:
            raise ValueError(f"port {n} not allowed (declare {PORT_MIN_DECL}-{PORT_MAX_DECL}, "
                             f"excluding {sorted(RESERVED_PORTS)})")
        if m.group(1) == "http":
            if http_port is not None:
                raise ValueError("only one http:N entry allowed")
            http_port = n
        elif m.group(1) == "tcp":
            tcp.add(n)
        else:
            udp.add(n)
        norm.append(f"{m.group(1)}:{n}")
    declared = tcp | udp | ({http_port} if http_port else set())
    return {"serve": not declared, "http": http_port, "tcp": tcp, "udp": udp,
            "declared": declared, "norm": norm}


def _build_cmd(pspec, wasm, serve_port: int, mem_bytes: int):
    """The wasmtime invocation for a ports spec. Returns (cmd, host_port, wait_ports).

    serve mode: `wasmtime serve` owns the one HTTP listener; no sockets granted.
    run mode:   `wasmtime run` with wasi:sockets granted (-Stcp/-Sudp/
                -Sinherit-network/-Sallow-ip-name-lookup, verified against
                wasmtime 45) so the app binds its declared ports itself. The
                grant is coarse (wasmtime can't allowlist per port), so the
                audit sweep below enforces the per-port firewall: bind an
                undeclared low port and the app is killed."""
    if pspec["serve"]:
        return ([WASMTIME, "serve", "-Scli", "-Shttp",
                 "-W", f"max-memory-size={mem_bytes}",
                 "--addr", f"{HOST_IP}:{serve_port}", str(wasm)],
                serve_port, [serve_port])
    cmd = [WASMTIME, "run", "-Scli", "-Stcp", "-Sudp",
           "-Sinherit-network", "-Sallow-ip-name-lookup",
           "-W", f"max-memory-size={mem_bytes}",
           "--env", "NAN_PORTS=" + ",".join(pspec["norm"]), str(wasm)]
    host_port = pspec["http"] or 0
    wait = [pspec["http"]] if pspec["http"] else sorted(pspec["tcp"])[:1]  # udp-only: no waitable port
    return cmd, host_port, wait


def launch(app_ref: str, name: str, share: float, mem_mb: int, pspec=None) -> dict:
    pspec = pspec or _parse_ports([])
    vid = "app_" + uuid.uuid4().hex[:9]
    port = _free_port() if pspec["serve"] else (pspec["http"] or 0)
    log_path = LOG_DIR / f"{vid}.log"
    rec = {"id": vid, "name": name or vid, "app": app_ref, "share": share,
           "hostPort": port, "sshHostPort": 0,
           "endpoint": f"http://{HOST_IP}:{port}" if port else None, "status": "starting",
           "createdAt": time.time(), "_proc": None, "_log": str(log_path),
           "error": None, "mem_mb": mem_mb,
           "ports": pspec["norm"], "boundPorts": [],
           "_declared": pspec["declared"],
           # what the audit allows: declared ports + the manager-assigned serve port
           "_allowed_ports": pspec["declared"] | ({port} if port else set())}
    with _lock:
        _apps[vid] = rec

    if MOCK:
        # Stand up a trivial responder so the full supervisor path is testable.
        mock_port = port or _free_port()
        rec["hostPort"] = mock_port
        rec["_proc"] = _mock_server(mock_port, vid)
        rec["status"] = "running"
        return rec

    try:
        wasm = _resolve_wasm(app_ref)
    except ValueError as e:
        rec["status"], rec["error"] = "failed", str(e)
        return rec

    # Default grants are minimal: no --dir (no fs), no --env beyond NAN_PORTS,
    # and sockets only when the version's firewall config asks for them.
    # `-W max-memory-size` caps the guest's linear memory (the only RAM a tenant
    # can grow) -- this is the real per-app memory ceiling, enforced by the
    # runtime rather than by RLIMIT_AS (see _preexec for why RLIMIT_AS is wrong).
    mem_bytes = max(mem_mb, 1) * 1024 * 1024
    cmd, host_port, wait_ports = _build_cmd(pspec, wasm, port, mem_bytes)
    rec["hostPort"] = host_port
    rec["endpoint"] = f"http://{HOST_IP}:{host_port}" if host_port else None
    logf = open(log_path, "wb")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf,
                                stderr=logf, preexec_fn=_preexec)
    except Exception as e:
        rec["status"], rec["error"] = "failed", f"spawn: {e}"
        logf.close()
        return rec
    rec["_proc"] = proc

    # readiness: a waitable port accepts, or the process dies first.
    # (udp-only apps have no waitable port: a short grace, then alive == running.)
    deadline = time.time() + (READY_SECS if wait_ports else 2.0)
    while time.time() < deadline:
        if proc.poll() is not None:
            rec["status"] = "failed"
            rec["error"] = _log_tail(log_path) or f"wasmtime exited {proc.returncode}"
            return rec
        if wait_ports and _port_open(wait_ports[0]):
            rec["status"] = "running"
            _audit_rec(rec)          # populate boundPorts right away (the bridge checks it)
            return rec
        time.sleep(0.1)
    if wait_ports:
        # timed out: keep it but flag; supervisor can decide
        rec["status"] = "running" if _port_open(wait_ports[0]) else "failed"
    else:
        rec["status"] = "running" if proc.poll() is None else "failed"
    if rec["status"] == "failed":
        rec["error"] = rec.get("error") or ("did not open port in time; " + (_log_tail(log_path) or ""))
        _kill(rec)
    else:
        _audit_rec(rec)
    return rec


# --- firewall enforcement: audit what each app actually bound ---------------- #
def _bound_ports(pid) -> set:
    """Ports bound by `pid`: its socket inodes (/proc/<pid>/fd) matched against
    /proc/net/{tcp,tcp6,udp,udp6}. TCP counts only LISTEN (st=0A); UDP counts
    unconnected binds. Unprivileged: the manager spawned these processes."""
    inodes = set()
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                ln = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if ln.startswith("socket:["):
                inodes.add(ln[8:-1])
    except OSError:
        return set()
    ports = set()
    for name in ("tcp", "tcp6", "udp", "udp6"):
        try:
            lines = pathlib.Path(f"/proc/net/{name}").read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) < 10 or f[9] not in inodes:
                continue
            if name.startswith("tcp") and f[3] != "0A":     # LISTEN only
                continue
            ports.add(int(f[1].rsplit(":", 1)[1], 16))
    return ports


def _audit_rec(rec):
    """Enforce the per-port firewall on one app. The wasmtime sockets grant is
    all-or-nothing, so this is the fine-grained half: any bind in the policed
    space (<= PORT_MAX_DECL, or reserved) that wasn't declared kills the app.
    Ephemeral outbound ports (32768+) are out of scope on purpose."""
    proc = rec.get("_proc")
    pid = getattr(proc, "pid", None)
    if pid is None or (hasattr(proc, "poll") and proc.poll() is not None):
        return
    bound = _bound_ports(pid)
    rec["boundPorts"] = sorted(bound & rec.get("_declared", set()))  # what the bridge may target
    policed = {p for p in bound if p <= PORT_MAX_DECL or p in RESERVED_PORTS}
    extra = policed - set(rec.get("_allowed_ports") or [])
    if extra:
        rec["status"] = "failed"
        rec["error"] = f"firewall: bound undeclared port(s) {sorted(extra)}; app killed"
        print(f"[audit] {rec['id']} killed: undeclared ports {sorted(extra)}", flush=True)
        _kill(rec)


def _audit_sweep():
    while True:
        time.sleep(AUDIT_SECS)
        with _lock:
            recs = [r for r in _apps.values() if r["status"] == "running"]
        for r in recs:
            try:
                _audit_rec(r)
            except Exception:
                pass


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
        try:
            pspec = _parse_ports(b.get("ports") or [])
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        # declared ports are first-come-first-served on shared loopback: reject a
        # launch whose declared ports collide with another live tenant's.
        if pspec["declared"]:
            with _lock:
                taken = set()
                for r in _apps.values():
                    if r["status"] in ("starting", "running"):
                        taken |= set(r.get("_declared") or set())
            clash = pspec["declared"] & taken
            if clash:
                return self._json(409, {"error": f"port(s) {sorted(clash)} already claimed by another tenant"})
        cat = _load_catalog()
        mem_mb = int(cat.get(app_ref, {}).get("mem_mb", DEF_MEM_MB))
        rec = launch(app_ref, b.get("name", ""), share, mem_mb, pspec)
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
    threading.Thread(target=_audit_sweep, daemon=True).start()   # firewall bind audit
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
