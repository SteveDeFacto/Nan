#!/usr/bin/env python3
"""runsc-manager: per-tenant gVisor (runsc / systrap) container manager.

Drop-in replacement for vmmanager on Tinfoil, where nested virtualization is
unavailable (no /dev/kvm, confirmed) so the qemu microVM path is dead. Each
tenant container runs under `runsc --platform=systrap` instead. It speaks the
SAME HTTP contract as vmmanager, so the supervisor's existing "vm" provision
backend talks to it with no change:

  POST   /vms   {image, share, name?, appPort?, gpu?, sshKey?}
                 -> 201 {id, status, endpoint, hostPort, sshHostPort, ...}
  DELETE /vms/:id        -> {id, deleted: true}
  GET    /vms/:id | /vms | /health | /capacity

ISOLATION MODEL (the whole point):
Every tenant runs in its OWN network namespace, connected to the host by a
dedicated veth pair on a private /30. gVisor's sandbox netstack attaches to the
tenant-side veth, so the container's services live at the tenant veth IP and
tenants cannot see each other's loopback or traffic. A small per-tenant TCP
forwarder bridges a unique HOST loopback port into the tenant (appPort and
ssh :22), which is exactly what the supervisor proxies via /x/:id and the SSH
bridge. No tenant shares a network namespace with another or with the host.

Reuses oci2microvm for image pull/unpack + busybox/dropbear/authorized_keys
injection; runsc consumes the rootfs directory directly (no ext4, no qemu).

Tenant egress is OFF by default (no default route, no NAT): inbound only, which
is the safer default for confidential multi-tenant compute. Set
RUNSC_TENANT_EGRESS=1 to give containers outbound network.
"""
import http.server
import ipaddress
import json
import os
import pathlib
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import oci2microvm as oci

# ---- config ---------------------------------------------------------------- #
PORT          = int(os.environ.get("RUNSC_MANAGER_PORT", "8091"))   # same port vmmanager used
NODE_VCPUS    = int(os.environ.get("NODE_VCPUS", "16"))
NODE_RAM_GB   = int(os.environ.get("NODE_RAM_GB", "64"))
RUNSC         = os.environ.get("RUNSC_BIN", "runsc")
RUNSC_PLATFORM= os.environ.get("RUNSC_PLATFORM", "systrap")
RUNSC_ROOT    = os.environ.get("RUNSC_ROOT", "/run/nan-runsc")
NET_BASE      = os.environ.get("RUNSC_NET_BASE", "10.123.0.0/16")    # carved into /30s, one per tenant
APPLY_LIMITS  = os.environ.get("RUNSC_APPLY_LIMITS", "1") not in ("", "0", "false")
TENANT_EGRESS = os.environ.get("RUNSC_TENANT_EGRESS", "") not in ("", "0", "false")
GPU_FORWARDING= os.environ.get("NAN_GPU_FORWARDING", "") not in ("", "0", "false")
MOCK          = os.environ.get("RUNSC_MOCK", "") not in ("", "0", "false")
LOG_DIR       = pathlib.Path(os.environ.get("RUNSC_LOG_DIR", "/tmp/nan-runsc-logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_containers: dict = {}
_used_share = 0.0
_next_idx = 0
_free_idx: list = []


# ---- allocation ------------------------------------------------------------ #
def _new_id() -> str:
    return "c" + secrets.token_hex(4)          # 9 chars; keeps veth ifnames < 15


def _alloc_idx() -> int:
    global _next_idx
    if _free_idx:
        return _free_idx.pop()
    i = _next_idx
    _next_idx += 1
    return i


def _free_index(i: int) -> None:
    if i is not None:
        _free_idx.append(i)


def _idx_subnet(idx: int):
    """One /30 per tenant: .1 host side, .2 container side. Distinct connected
    routes in the host netns, so no cross-tenant overlap."""
    net = ipaddress.ip_network(NET_BASE)
    base = int(net.network_address) + idx * 4
    return str(ipaddress.ip_address(base + 1)), str(ipaddress.ip_address(base + 2)), 30


def _alloc_host_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---- networking ------------------------------------------------------------ #
def _run(cmd, check=True):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} :: {(p.stderr or p.stdout).strip()}")
    return p


def setup_netns(vid: str, idx: int):
    """Create an UNNAMED net namespace for the tenant, held open by a placeholder
    process, plus a veth pair. Returns (holder, nspath, host_ip, cont_ip, hveth).

    We avoid `ip netns add` on purpose: it bind-mounts /run/netns and marks it a
    shared mount, which the enclave's locked mount propagation refuses (EPERM)
    even though we hold CAP_SYS_ADMIN. `unshare --net` needs no mount setup, so
    the namespace is referenced by /proc/<pid>/ns/net and configured with nsenter
    instead of `ip -n <name>`. Uses only CAP_NET_ADMIN + CAP_SYS_ADMIN (setns),
    both of which the container has."""
    hveth = f"nh{vid[1:]}"          # 'nh' + 8 hex = 10 chars (< 15 ifname limit)
    cveth = f"nc{vid[1:]}"
    host_ip, cont_ip, prefix = _idx_subnet(idx)

    # Placeholder process owning a fresh net namespace (kept alive until teardown).
    holder = subprocess.Popen(["unshare", "--net", "--", "sleep", "infinity"],
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = holder.pid
    nspath = f"/proc/{pid}/ns/net"
    for _ in range(100):
        if os.path.exists(nspath):
            break
        if holder.poll() is not None:
            raise RuntimeError("netns placeholder (unshare --net) exited; unshare may be blocked")
        time.sleep(0.02)
    # Isolation is the whole point: refuse to proceed if the namespace is not
    # genuinely separate from the manager's (a silently-shared netns would run
    # tenants with NO network isolation). Fail the deployment instead.
    try:
        if os.readlink(nspath) == os.readlink("/proc/self/ns/net"):
            holder.kill()
            raise RuntimeError("tenant netns is not isolated (same inode as manager); "
                               "unshare --net did not create a new namespace")
    except OSError as e:
        holder.kill()
        raise RuntimeError(f"could not verify tenant netns isolation: {e}")

    _run(["ip", "link", "add", hveth, "type", "veth", "peer", "name", cveth])
    _run(["ip", "link", "set", cveth, "netns", str(pid)])        # move container end into the ns by PID
    _run(["ip", "addr", "add", f"{host_ip}/{prefix}", "dev", hveth])
    _run(["ip", "link", "set", hveth, "up"])
    ns = ["nsenter", "-t", str(pid), "-n"]                       # run `ip` inside the tenant netns
    _run(ns + ["ip", "addr", "add", f"{cont_ip}/{prefix}", "dev", cveth])
    _run(ns + ["ip", "link", "set", cveth, "up"])
    _run(ns + ["ip", "link", "set", "lo", "up"])
    if TENANT_EGRESS:
        _run(ns + ["ip", "route", "add", "default", "via", host_ip], check=False)
    return holder, nspath, host_ip, cont_ip, hveth


def teardown_netns(holder, hveth: str):
    if holder is not None:
        try:
            holder.kill()          # reaps the netns; the veth pair dies with it
        except Exception:
            pass
    if hveth:
        _run(["ip", "link", "del", hveth], check=False)          # usually already gone with the ns


class Forwarder:
    """Threaded TCP forwarder: 127.0.0.1:listen_port -> dst_host:dst_port.
    Runs in the manager (host) netns, dials the tenant veth IP. One per tenant
    port (app + ssh)."""
    def __init__(self, listen_port, dst_host, dst_port):
        self.listen_port, self.dst_host, self.dst_port = listen_port, dst_host, dst_port
        self.srv = None
        self.alive = False

    def start(self):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", self.listen_port))
        self.srv.listen(64)
        self.srv.settimeout(0.5)            # so stop() is noticed even with no traffic
        self.alive = True
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while self.alive:
            try:
                client, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(None)         # long-lived (e.g. SSH) sessions must not inherit the timeout
            threading.Thread(target=self._pipe, args=(client,), daemon=True).start()
        try:
            self.srv.close()
        except OSError:
            pass

    def _pipe(self, client):
        try:
            upstream = socket.create_connection((self.dst_host, self.dst_port), timeout=10)
        except OSError:
            client.close()
            return

        def shovel(a, b):
            try:
                while True:
                    data = a.recv(65536)
                    if not data:
                        break
                    b.sendall(data)
            except OSError:
                pass
            finally:
                for s in (a, b):
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        threading.Thread(target=shovel, args=(client, upstream), daemon=True).start()
        shovel(upstream, client)
        client.close()
        upstream.close()

    def stop(self):
        self.alive = False
        try:
            self.srv.close()
        except OSError:
            pass

# ---- OCI bundle ------------------------------------------------------------ #
def oci_spec(rootfs: pathlib.Path, netns_path: str, res: dict) -> dict:
    caps = ["CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_FOWNER", "CAP_SETUID", "CAP_SETGID",
            "CAP_NET_BIND_SERVICE", "CAP_KILL"]
    spec = {
        "ociVersion": "1.0.0",
        "process": {
            "terminal": False,
            "user": {"uid": 0, "gid": 0},
            "args": ["/nan-init"],           # inject_init wrote this; it starts dropbear + the workload
            "env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME=/root"],
            "cwd": "/",
            "capabilities": {k: list(caps) for k in ("bounding", "effective", "permitted")},
            "rlimits": [{"type": "RLIMIT_NOFILE", "hard": 65536, "soft": 65536}],
        },
        "root": {"path": str(rootfs), "readonly": False},
        "hostname": "nan",
        "mounts": [
            {"destination": "/proc", "type": "proc", "source": "proc"},
            {"destination": "/dev", "type": "tmpfs", "source": "tmpfs",
             "options": ["nosuid", "strictatime", "mode=755", "size=65536k"]},
            {"destination": "/dev/pts", "type": "devpts", "source": "devpts",
             "options": ["nosuid", "noexec", "newinstance", "ptmxmode=0666", "mode=0620"]},
            {"destination": "/dev/shm", "type": "tmpfs", "source": "shm",
             "options": ["nosuid", "noexec", "nodev", "mode=1777", "size=65536k"]},
            {"destination": "/sys", "type": "sysfs", "source": "sysfs",
             "options": ["nosuid", "noexec", "nodev", "ro"]},
            {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs",
             "options": ["nosuid", "nodev", "mode=1777"]},
        ],
        "linux": {
            "namespaces": [
                {"type": "pid"}, {"type": "ipc"}, {"type": "uts"}, {"type": "mount"},
                {"type": "network", "path": netns_path},
            ],
        },
    }
    if APPLY_LIMITS:
        spec["linux"]["resources"] = {
            "memory": {"limit": res["ram_mib"] * 1024 * 1024},
            "cpu": {"quota": res["vcpus"] * 100000, "period": 100000},
        }
    return spec


def build_bundle(image: str, work: pathlib.Path, ssh_key, netns_path: str, res: dict) -> pathlib.Path:
    bundle = oci.pull_and_unpack(image, work)          # bundle/ with rootfs/ + umoci config.json
    cfg = oci.read_image_config(bundle)                # read BEFORE overwriting config.json
    rootfs = bundle / "rootfs"
    oci.inject_init(rootfs, cfg, oci.GUEST_BUSYBOX, ssh_key=ssh_key)
    (bundle / "config.json").write_text(json.dumps(oci_spec(rootfs, netns_path, res), indent=2))
    return bundle


# ---- lifecycle ------------------------------------------------------------- #
def _capacity() -> dict:
    free = max(0.0, 1.0 - _used_share)
    return {"maxShare": round(free, 4), "usedShare": round(_used_share, 4),
            "vcpusFree": round(free * NODE_VCPUS, 1), "ramGbFree": round(free * NODE_RAM_GB, 1)}


def _public(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def _spawn(image: str, share: float, name: str, app_port: int, gpu: bool, ssh_key=None) -> dict:
    global _used_share
    vid = _new_id()
    idx = _alloc_idx()
    host_port = _alloc_host_port()
    ssh_host_port = _alloc_host_port()
    res = oci.derive_resources(share)
    log_path = str(LOG_DIR / f"{vid}.log")
    rec = {
        "id": vid, "name": name or vid, "image": image,
        "share": res["share"], "pct": res["pct"], "vcpus": res["vcpus"],
        "ramMib": res["ram_mib"], "gpu": gpu, "appPort": app_port,
        "hostPort": host_port, "sshHostPort": ssh_host_port,
        "endpoint": f"http://127.0.0.1:{host_port}",
        "status": "provisioning", "createdAt": time.time(), "error": None,
        "_idx": idx, "_ns_holder": None, "_hveth": None, "_proc": None, "_fwd": [], "_log": log_path,
        "_stopping": False,
    }
    with _lock:
        _containers[vid] = rec
        _used_share += res["share"]

    def worker():
        try:
            work = pathlib.Path(tempfile.mkdtemp(prefix=f"nanrunsc-{vid}-"))
            rec["status"] = "building"
            if MOCK:
                # Contract/threading test without root/runsc/registry: simulate a
                # running container with a trivial loopback echo target.
                tgt = _alloc_host_port()
                _mock_target(tgt, log_path)
                fa = Forwarder(host_port, "127.0.0.1", tgt); fa.start()
                fs = Forwarder(ssh_host_port, "127.0.0.1", tgt); fs.start()
                rec["_fwd"] = [fa, fs]
                rec["status"] = "running"
                while not rec["_stopping"]:
                    time.sleep(0.2)
                rec["status"] = "stopped"
                return

            holder, nspath, host_ip, cont_ip, hveth = setup_netns(vid, idx)
            rec["_ns_holder"], rec["_hveth"] = holder, hveth
            bundle = build_bundle(image, work, ssh_key, nspath, res)
            fa = Forwarder(host_port, cont_ip, app_port); fa.start()
            fs = Forwarder(ssh_host_port, cont_ip, 22); fs.start()
            rec["_fwd"] = [fa, fs]
            cmd = [RUNSC, f"--platform={RUNSC_PLATFORM}", "--network=sandbox", f"--root={RUNSC_ROOT}"]
            if not APPLY_LIMITS:
                cmd.append("--ignore-cgroups")
            cmd += ["run", "--bundle", str(bundle), vid]
            rec["status"] = "booting"
            with open(log_path, "wb") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL)
            rec["_proc"] = proc
            rec["status"] = "running"
            proc.wait()
            if rec["status"] == "running":
                rec["status"] = "stopped"
        except Exception as e:                       # noqa: BLE001 - surface to the API
            rec["status"] = "failed"
            rec["error"] = str(e)
        finally:
            _cleanup(rec)

    threading.Thread(target=worker, daemon=True).start()
    return rec


def _mock_target(port, log_path):
    def serve():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port)); s.listen(8)
        with open(log_path, "a") as lf:
            lf.write(f"[mock] target up on 127.0.0.1:{port}\n")
        while True:
            try:
                c, _ = s.accept()
            except OSError:
                break
            c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nnan\n")
            c.close()
    threading.Thread(target=serve, daemon=True).start()


def _cleanup(rec: dict):
    for f in rec.get("_fwd", []):
        try:
            f.stop()
        except Exception:
            pass
    if not MOCK:
        _run([RUNSC, f"--root={RUNSC_ROOT}", "delete", "--force", rec["id"]], check=False)
        teardown_netns(rec.get("_ns_holder"), rec.get("_hveth"))
    with _lock:
        global _used_share
        _used_share = max(0.0, _used_share - rec["share"])
        _free_index(rec.get("_idx"))


def _kill(vid: str) -> bool:
    rec = _containers.get(vid)
    if not rec:
        return False
    rec["_stopping"] = True
    if not MOCK:
        _run([RUNSC, f"--root={RUNSC_ROOT}", "kill", vid, "SIGKILL"], check=False)
        proc = rec.get("_proc")
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    return True


# ---- HTTP ------------------------------------------------------------------ #
def _debug_env() -> dict:
    """Report, from the manager's ACTUAL runtime, the facts that decide the
    isolation design: kernel string (gVisor leaves a signature), whether any
    unshare variant makes a genuinely distinct netns, and whether nested runsc
    runs here at all. This is read-only."""
    out = {}
    try:
        out["uname"] = " ".join(os.uname())
    except Exception as e:
        out["uname"] = f"err: {e}"
    try:
        out["proc_version"] = open("/proc/version").read().strip()
    except Exception as e:
        out["proc_version"] = f"err: {e}"
    try:
        out["init_netns"] = os.readlink("/proc/self/ns/net")
    except Exception as e:
        out["init_netns"] = f"err: {e}"

    variants = {"net": ["--net"],
                "net_user": ["--net", "--user", "--map-root-user"],
                "user_net": ["--user", "--map-root-user", "--net"]}
    netns = {}
    for name, flags in variants.items():
        try:
            h = subprocess.Popen(["unshare", *flags, "--", "sleep", "10"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE)
            p = f"/proc/{h.pid}/ns/net"
            for _ in range(50):
                if os.path.exists(p) or h.poll() is not None:
                    break
                time.sleep(0.02)
            if h.poll() is not None:
                netns[name] = {"distinct": False,
                               "err": (h.stderr.read().decode().strip() or "exited")[:160]}
            else:
                netns[name] = {"distinct": os.readlink(p) != os.readlink("/proc/self/ns/net")}
                h.kill()
        except Exception as e:
            netns[name] = {"error": str(e)}
    out["netns"] = netns

    try:
        v = subprocess.run([RUNSC, "--version"], capture_output=True, text=True, timeout=10)
        lines = (v.stdout or v.stderr or "").strip().splitlines()
        out["runsc_version"] = lines[0] if lines else ""
    except Exception as e:
        out["runsc_version"] = f"err: {e}"
    try:
        r = subprocess.run([RUNSC, "--platform=systrap", "--network=none",
                            f"--root={RUNSC_ROOT}", "do", "/bin/echo", "nan-ok"],
                           capture_output=True, text=True, timeout=30)
        out["runsc_do"] = {"ok": "nan-ok" in (r.stdout or ""),
                           "stderr": (r.stderr or "").strip()[-500:]}
    except Exception as e:
        out["runsc_do"] = {"ok": False, "error": str(e)}
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"ok": True, "runtime": "runsc", "platform": RUNSC_PLATFORM,
                                    "mock": MOCK, "capacity": _capacity()})
        if self.path == "/debug/env":
            return self._json(200, _debug_env())
        if self.path == "/capacity":
            return self._json(200, _capacity())
        if self.path == "/vms":
            with _lock:
                return self._json(200, {"vms": [_public(r) for r in _containers.values()]})
        if self.path.startswith("/vms/"):
            rec = _containers.get(self.path.split("/", 2)[2])
            return self._json(200, _public(rec)) if rec else self._json(404, {"error": "not found"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/vms":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad json"})
        image = req.get("image")
        share = req.get("share")
        gpu = bool(req.get("gpu", False))
        app_port = int(req.get("appPort", 8080))
        ssh_key = req.get("sshKey") or None
        if not image or not isinstance(share, (int, float)):
            return self._json(422, {"error": "image (str) and share (0..1) required"})
        if not (0 < share <= 1):
            return self._json(422, {"error": "share must be in (0, 1]"})
        if gpu and not GPU_FORWARDING:
            return self._json(501, {"error": "gpu_forwarding_unavailable",
                                    "message": "GPU passthrough under gVisor (nvproxy) is not wired yet. "
                                               "Request gpu=false."})
        with _lock:
            if share > (1.0 - _used_share) + 1e-9:
                return self._json(409, {"error": "not enough free share", "capacity": _capacity()})
        rec = _spawn(image, float(share), req.get("name", ""), app_port, gpu, ssh_key=ssh_key)
        return self._json(201, _public(rec))

    def do_DELETE(self):
        if self.path.startswith("/vms/"):
            vid = self.path.split("/", 2)[2]
            ok = _kill(vid)
            return self._json(200 if ok else 404, {"id": vid, "deleted": ok})
        return self._json(404, {"error": "not found"})


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    pathlib.Path(RUNSC_ROOT).mkdir(parents=True, exist_ok=True)
    srv = Server(("0.0.0.0", PORT), Handler)
    print(f"[runsc-manager] listening on :{PORT}  platform={RUNSC_PLATFORM}  "
          f"mock={MOCK}  limits={APPLY_LIMITS}  egress={TENANT_EGRESS}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
