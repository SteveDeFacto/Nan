#!/usr/bin/env python3
"""
NAN GPU worker — Layer 2 (per-tenant MPS-capped processes).

Two roles in one file:

  MANAGER (default).  No CUDA context of its own — a pure control plane. Forks
  one CHILD process per tenant, each launched with that tenant's MPS cap in its
  environment, and proxies submissions to the right child. Tracks lifecycle and
  share capacity. Because the manager never imports cupy, it runs (and is
  testable) anywhere; only children touch the GPU.

  CHILD  (`worker.py child`).  Sets up CUDA AFTER inheriting
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE from the manager, so the MPS daemon applies
  the SM cap to this process. Holds its own context (separate GPU address space
  from other tenants), reports the SM count the driver granted it (the cap proof
  == the validated %smid finding), and runs submitted PTX.

ISOLATION — what Layer 2 gives and what it does NOT (no faking):
  GIVES: process-level separation. Each tenant is a distinct OS process with a
    distinct CUDA context / GPU virtual address space, so one tenant cannot
    address another's memory through ordinary pointers, and an MPS SM cap is
    enforced per process. Teardown frees the child's context back to the driver.
  DOES NOT: zero freed VRAM, or stop a malicious kernel from probing physical
    memory it was handed. Residual-data protection and out-of-bounds fencing are
    Layer 4 (fence_ptx stub + the adversarial probe). Until then children run
    ONLY trusted PTX (REQUIRE_FENCE gate). <-- do not weaken this lightly.
"""
import os, sys, json, time, base64, threading, subprocess, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# -------- shared config (both roles) ----------------------------------------
PORT        = int(os.environ.get("WORKER_PORT", "8090"))   # manager port (child overrides via env)
VRAM_GB     = float(os.environ.get("GPU_VRAM_GB", "141"))
NODE_VCPUS  = int(os.environ.get("NODE_VCPUS", "16"))
CHILD_BASE  = int(os.environ.get("CHILD_PORT_BASE", "8100"))
CUDA_ATTR_MULTIPROCESSOR_COUNT = 16   # cudaDevAttrMultiProcessorCount — reflects the MPS cap
REQUIRE_FENCE = os.environ.get("REQUIRE_FENCE", "1") not in ("0", "false", "off")


def _http_json(method, url, payload=None, timeout=5):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read() or b"{}")


# ===========================================================================
#  CHILD  — one tenant, one MPS-capped CUDA context
# ===========================================================================
def run_child():
    # Import cupy HERE (child only) so the manager stays GPU-free. A hard import
    # failure crashes the child fast instead of being retried as a transient.
    import cupy
    from cupy.cuda import runtime, function

    mps_cap = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")  # set by the manager
    cport   = int(os.environ.get("WORKER_PORT", "8090"))
    state   = {"ready": False, "device": "?", "sm_granted": -1}

    def init_cuda(retries=30, gap=2.0):
        """Create the capped context. Retries because the MPS daemon may still be
        coming up (boot race). The cap was set in env BEFORE this runs."""
        last = None
        for _ in range(retries):
            try:
                cupy.cuda.Device(0).use()
                _ = cupy.zeros(1)                    # force primary-context creation here
                props = runtime.getDeviceProperties(0)
                name = props["name"]
                state["device"] = name.decode() if isinstance(name, bytes) else str(name)
                # attribute query (NOT props['multiProcessorCount'] which is the static
                # physical count) — this reflects the MPS ACTIVE_THREAD_PERCENTAGE cap.
                state["sm_granted"] = runtime.deviceGetAttribute(CUDA_ATTR_MULTIPROCESSOR_COUNT, 0)
                state["ready"] = True
                print(f"[child:{cport}] CUDA up: {state['device']} | cap={mps_cap or 'none'}% "
                      f"| SMs granted={state['sm_granted']}", flush=True)
                return True
            except Exception as e:                   # noqa: BLE001
                last = e; time.sleep(gap)
        print(f"[child:{cport}] CUDA init failed: {last}", flush=True)
        return False

    def fence_ptx(ptx: bytes) -> bytes:
        """SAFETY STUB (Layer 4): untrusted PTX must have every global load/store
        masked to the tenant's own allocation before running. Until real, refuse."""
        raise NotImplementedError("PTX bounds-fencing not implemented (Layer 4)")

    def run_ptx(job: dict) -> dict:
        ptx = base64.b64decode(job["ptx_b64"])
        if REQUIRE_FENCE and not job.get("trusted"):
            ptx = fence_ptx(ptx)                     # raises until Layer 4
        entry     = job["entry"]
        grid      = tuple(job.get("grid",  [1, 1, 1]))
        block     = tuple(job.get("block", [1, 1, 1]))
        out_bytes = int(job["out_bytes"])
        mod = function.Module(); mod.load(ptx)       # cuModuleLoadData on raw PTX
        fn = mod.get_function(entry)
        out = cupy.zeros(out_bytes, dtype=cupy.uint8)
        fn(grid, block, (out,))
        runtime.deviceSynchronize()
        return {"ok": True, "output_b64": base64.b64encode(cupy.asnumpy(out).tobytes()).decode()}

    class CH(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path in ("/health", "/healthz"):
                self._send(200 if state["ready"] else 503, {
                    "ok": state["ready"], "device": state["device"],
                    "mps_cap_pct": mps_cap, "sm_granted": state["sm_granted"], "vram_gb": VRAM_GB})
            else: self._send(404, {"error": "not_found"})
        def do_POST(self):
            if self.path != "/run": return self._send(404, {"error": "not_found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                self._send(200, run_ptx(json.loads(self.rfile.read(n) or b"{}")))
            except NotImplementedError as e:
                self._send(403, {"ok": False, "error": "unfenced_ptx_refused", "detail": str(e)})
            except Exception as e:                   # noqa: BLE001
                self._send(400, {"ok": False, "error": str(e)})

    if not init_cuda():
        raise SystemExit("child: no CUDA context (MPS sidecar up? GPU attached?)")
    print(f"[child:{cport}] listening", flush=True)
    ThreadingHTTPServer(("127.0.0.1", cport), CH).serve_forever()


# ===========================================================================
#  MANAGER — forks/cap/tracks/proxies per-tenant children (no CUDA here)
# ===========================================================================
_lock = threading.Lock()
_tenants: dict[str, dict] = {}     # id -> rec
_used_share = 0.0
_next_port = CHILD_BASE


def _alloc_port():
    global _next_port
    p = _next_port; _next_port += 1; return p


def _pub(rec):  # strip the Popen handle
    return {k: v for k, v in rec.items() if k != "_proc"}


def _spawn_tenant(tid: str, share: float) -> dict:
    global _used_share
    pct  = max(1, round(share * 100))
    port = _alloc_port()
    env  = {**os.environ,
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(pct),   # the cap, set BEFORE the child's context
            "WORKER_PORT": str(port),
            "REQUIRE_FENCE": "1" if REQUIRE_FENCE else "0"}
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "child"], env=env)
    rec = {"id": tid, "share": share, "pct": pct, "port": port, "status": "starting",
           "sm_granted": None, "device": None, "error": None, "createdAt": time.time(), "_proc": proc}
    with _lock:
        _tenants[tid] = rec
        _used_share += share

    # wait for the child to report healthy (or die), so the API answer carries the proof
    deadline = time.time() + 75
    while time.time() < deadline:
        if proc.poll() is not None:                  # child exited before becoming ready
            rec["status"] = "failed"
            rec["error"] = f"child exited rc={proc.returncode} (no GPU/MPS? cupy missing?)"
            with _lock: _used_share = max(0.0, _used_share - share)
            return rec
        try:
            code, h = _http_json("GET", f"http://127.0.0.1:{port}/health", timeout=2)
            if code == 200 and h.get("ok"):
                rec.update(status="running", sm_granted=h.get("sm_granted"), device=h.get("device"))
                return rec
        except Exception:                            # noqa: BLE001 — not up yet
            pass
        time.sleep(1.0)
    rec["status"] = "timeout"; rec["error"] = "child did not become healthy in time"
    return rec


def _kill_tenant(tid: str) -> bool:
    global _used_share
    rec = _tenants.get(tid)
    if not rec: return False
    proc = rec.get("_proc")
    if proc and proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=5)
        except Exception: proc.kill()               # noqa: BLE001
    rec["status"] = "stopped"
    with _lock: _used_share = max(0.0, _used_share - rec["share"])
    return True


def _capacity():
    free = max(0.0, 1.0 - _used_share)
    return {"maxShare": round(free, 4), "usedShare": round(_used_share, 4),
            "smFree": round(free * 132), "vramFreeGb": round(free * VRAM_GB, 1)}


class MGR(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            return self._send(200, {"ok": True, "role": "manager",
                                    "mps_pipe": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
                                    "tenants": [_pub(r) for r in _tenants.values()],
                                    "capacity": _capacity()})
        if self.path == "/tenants":
            return self._send(200, {"tenants": [_pub(r) for r in _tenants.values()]})
        if self.path.startswith("/tenants/"):
            tid = self.path.split("/", 2)[2]
            rec = _tenants.get(tid)
            if not rec: return self._send(404, {"error": "no such tenant"})
            # live child health (fresh sm_granted) if running
            if rec.get("_proc") and rec["_proc"].poll() is None:
                try:
                    _, h = _http_json("GET", f"http://127.0.0.1:{rec['port']}/health", timeout=2)
                    rec.update(sm_granted=h.get("sm_granted"), device=h.get("device"))
                except Exception: pass               # noqa: BLE001
            return self._send(200, _pub(rec))
        return self._send(404, {"error": "not_found"})

    def do_POST(self):
        # create a capped tenant worker
        if self.path == "/tenants":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._send(400, {"error": "bad json"})
            share = req.get("share")
            if not isinstance(share, (int, float)) or not (0 < share <= 1):
                return self._send(422, {"error": "share must be in (0,1]"})
            with _lock:
                if share > (1.0 - _used_share) + 1e-9:
                    return self._send(409, {"error": "not enough free share", "capacity": _capacity()})
            tid = req.get("id") or ("t_" + base64.urlsafe_b64encode(os.urandom(4)).decode().rstrip("="))
            rec = _spawn_tenant(tid, float(share))
            return self._send(201 if rec["status"] == "running" else 502, _pub(rec))
        # proxy a submission to a tenant's child
        if self.path.startswith("/tenants/") and self.path.endswith("/run"):
            tid = self.path.split("/")[2]
            rec = _tenants.get(tid)
            if not rec or rec["status"] != "running":
                return self._send(409, {"error": "tenant not running"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                job = json.loads(self.rfile.read(n) or b"{}")
                code, out = _http_json("POST", f"http://127.0.0.1:{rec['port']}/run", job, timeout=120)
                return self._send(code, out)
            except urllib.error.HTTPError as e:
                return self._send(e.code, json.loads(e.read() or b"{}"))
            except Exception as e:                   # noqa: BLE001
                return self._send(502, {"error": f"child unreachable: {e}"})
        return self._send(404, {"error": "not_found"})

    def do_DELETE(self):
        if self.path.startswith("/tenants/"):
            tid = self.path.split("/", 2)[2]
            return self._send(200, {"id": tid, "status": "stopped"}) if _kill_tenant(tid) \
                else self._send(404, {"error": "no such tenant"})
        return self._send(404, {"error": "not_found"})


def run_manager():
    print(f"[manager] listening on :{PORT} | child ports from {CHILD_BASE} "
          f"| MPS pipe={os.environ.get('CUDA_MPS_PIPE_DIRECTORY')}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), MGR).serve_forever()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        run_child()
    else:
        run_manager()
