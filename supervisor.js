// NAN supervisor - the WHOLE service, running INSIDE the Tinfoil enclave behind
// the shim (the single ingress). It is the measured/attested image: the same
// published code that checks a user's signature, gates on escrow, mints the
// session token, launches the per-use container, and proxies the data path.
//
// There is no external tier. Browser -> shim -> here, for BOTH control and data:
//   control:  /v1/*        (SIWE login, deployments, account, attestation)
//   data:     /x/:id/*     (verify session token + ownership, proxy to the
//                           spawned container; fly.io used to do nothing here -
//                           now nothing external touches a prompt at all)
//
// One signing SECRET (an enclave secret). One token type: the session JWT the
// browser gets at login is reused as the capability on the data path.
//
// >>> The ONLY thing left to implement for your CVM is spawn/stop/measure below.

import express from "express";
import cors from "cors";
import http from "node:http";
import https from "node:https";
import net from "node:net";
import { createHash, randomBytes } from "node:crypto";
import { gunzipSync } from "node:zlib";
import { execFileSync, execFile } from "node:child_process";
import { promisify } from "node:util";
const pexec = promisify(execFile);
import { mkdtempSync, readFileSync, rmSync, mkdirSync, writeFileSync, renameSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { WebSocketServer } from "ws";
import { verifyMessage, createPublicClient, createWalletClient, http as viemHttp, getAddress, keccak256, toHex, stringToBytes } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { base } from "viem/chains";
import { SignJWT, jwtVerify } from "jose";

// ----------------------------------------------------------------------------
// config
// ----------------------------------------------------------------------------
const PORT           = parseInt(process.env.PORT || "8080", 10);
const SECRET         = new TextEncoder().encode(need("SECRET")); // signs + verifies the session/capability token
const PUBLIC_URL     = (process.env.PUBLIC_URL || "").replace(/\/+$/, ""); // own shim URL; else derived per-request
const SIWE_DOMAIN    = process.env.SIWE_DOMAIN || "nan.host";
const SIWE_URI       = process.env.SIWE_URI || "https://nan.host";
const CHAIN_ID       = parseInt(process.env.CHAIN_ID || "8453", 10);
const CORS_ORIGINS   = (process.env.CORS_ORIGINS || "https://nan.host").split(",").map(s => s.trim()).filter(Boolean);
// --- pay-per-deploy (no custody): users pay the NanPay forwarder; the supervisor
//     WATCHES it for Paid events and converts each payment to runtime. No held
//     balance, no escrow contract, no key in the enclave that can move funds.
const FORWARDER_ADDRESS  = process.env.FORWARDER_ADDRESS || "";   // NanPay contract (watch-only)
const USDC_ADDRESS       = process.env.USDC_ADDRESS || "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"; // USDC on Base
// --- app approval: NanAppCatalog (read-only) is the deploy gate for catalog apps.
//     Only the catalog's owner (the EOA that deployed it) can approve/reject a
//     version, by signing a setApproval transaction; an ipfs://<cid> deploy is
//     refused until its version is Approved. Empty = catalog apps can't deploy
//     at all (fail closed); baked-in app ids are exempt (attested image content).
const APP_CATALOG_ADDRESS = process.env.APP_CATALOG_ADDRESS || "";
const PAYMENT_WINDOW_SEC = parseInt(process.env.PAYMENT_WINDOW_SEC || "600", 10); // unpaid awaiting_payment TTL
const GRACE_SEC          = parseInt(process.env.GRACE_SEC || "90", 10);           // post-expiry grace before teardown
const PAY_POLL_SEC       = parseInt(process.env.PAY_POLL_SEC || "12", 10);        // Base log poll interval
// --- fair billing: funded runtime is a BALANCE, not a wall-clock deadline -----
// remainingMs drains only on ticks where the platform is actually serving:
// supervisor up, backend manager healthy, payment watcher fresh, app instance
// alive. Any outage FREEZES every clock; it resumes on the first healthy tick.
// State is persisted so a supervisor restart freezes (never forfeits) the clock.
const BILL_TICK_SEC      = parseInt(process.env.BILL_TICK_SEC || "15", 10);       // billing/reaper cadence
const WATCHER_STALE_SEC  = Math.max(60, 5 * PAY_POLL_SEC);                        // watcher silence that freezes billing
const STATE_FILE         = process.env.STATE_FILE || "/var/lib/nan/state.json";   // mount a volume here to survive restarts
// manual-billing / pilot: boot deployments WITHOUT waiting for an on-chain payment.
//   AUTO_PROVISION=1            -> every deploy provisions immediately (closed pilot).
//   ADMIN_TOKEN set            -> operator can provision one deployment on demand via
//                                 POST /v1/admin/deployments/:id/provision (x-admin-token).
//   AUTO_PROVISION_HOURS > 0   -> optional safety expiry; 0 = runs until deleted.
const AUTO_PROVISION       = /^(1|true|on)$/i.test(process.env.AUTO_PROVISION || "");
const AUTO_PROVISION_HOURS = parseFloat(process.env.AUTO_PROVISION_HOURS || "0");
const ADMIN_TOKEN          = process.env.ADMIN_TOKEN || "";
const BASE_RPC       = process.env.BASE_RPC || "https://mainnet.base.org";
const SESSION_TTL    = parseInt(process.env.SESSION_TTL || "43200", 10); // 12h: long enough to cover a deployment's data-path use
const SSH_USER       = process.env.SSH_USER || "instance"; // login user the supervisor's sshd drops into
const DEFAULT_IMAGE  = process.env.DEFAULT_IMAGE || "debian:bookworm-slim"; // any stock image; sshd is hosted by the supervisor, not the image
// --- worker launch (per-tenant container = the only isolation boundary) ------
const DOCKER_SOCK    = process.env.DOCKER_SOCK || "/var/run/docker.sock";  // Engine API endpoint (mounted into the supervisor)
const MPS_PIPE_DIR   = process.env.CUDA_MPS_PIPE_DIRECTORY || "/tmp/nvidia-mps";
const ENABLE_MPS     = !/^(0|false|off)$/i.test(process.env.ENABLE_MPS || "1"); // MPS enforces BOTH the SM cap and the VRAM cap (validated under CC)
const WORKER_PREFIX  = process.env.WORKER_PREFIX || "nan_";
const SPAWN_TIMEOUT_MS = parseInt(process.env.SPAWN_TIMEOUT_MS || "180000", 10); // includes image pull
const WORKER_MEM      = process.env.WORKER_MEM || "16g";                // host-RAM cap per worker (not GPU)
const WORKER_PIDS     = process.env.WORKER_PIDS || "512";
// ---- worker MANAGER (Layer 2/3) --------------------------------------------
// The GPU container runs a manager that forks one MPS-capped CHILD PROCESS per
// tenant. The supervisor routes deploys/submissions HERE instead of creating
// containers itself (Tinfoil forbids runtime container creation). Reachable over
// the enclave-local network; default loopback.
const WORKER_MGR_URL  = (process.env.WORKER_MGR_URL || "http://127.0.0.1:8090").replace(/\/+$/, "");
// provisioning backend: "worker" = GPU PTX submission (default), "vm" = tenant-app
// hosting via the app manager on VMMGR_URL (the wasm-manager runs each app as a
// `wasmtime serve` process). The "vm"/VMMGR_URL names are legacy, kept for config compat.
const PROVISION_BACKEND = (process.env.PROVISION_BACKEND || "worker").toLowerCase();
const VMMGR_URL = (process.env.VMMGR_URL || "http://127.0.0.1:8091").replace(/\/+$/, "");
function mgrReq(method, path, body, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    const u = new URL(WORKER_MGR_URL + path);
    const data = body != null ? Buffer.from(JSON.stringify(body)) : null;
    const r = http.request(
      { host: u.hostname, port: u.port || 80, path: u.pathname + u.search, method, timeout: timeoutMs,
        headers: { "Content-Type": "application/json", ...(data ? { "Content-Length": data.length } : {}) } },
      (res) => { let buf = ""; res.on("data", (c) => (buf += c));
                 res.on("end", () => { let j; try { j = JSON.parse(buf || "{}"); } catch { j = { raw: buf }; }
                                       resolve({ status: res.statusCode || 0, body: j }); }); });
    r.on("error", reject); r.on("timeout", () => r.destroy(new Error("manager timeout")));
    if (data) r.write(data); r.end();
  });
}
async function mgrHealth(timeoutMs = 3000) {
  const r = await mgrReq("GET", "/health", null, timeoutMs);
  if (r.status !== 200) throw new Error(`manager /health ${r.status}`);
  return r.body;
}

// --- app manager client ("vm" backend on VMMGR_URL; the wasm-manager) --------
function vmReq(method, path, body, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    const u = new URL(VMMGR_URL + path);
    const data = body != null ? Buffer.from(JSON.stringify(body)) : null;
    const r = http.request(
      { host: u.hostname, port: u.port || 80, path: u.pathname + u.search, method, timeout: timeoutMs,
        headers: { "Content-Type": "application/json", ...(data ? { "Content-Length": data.length } : {}) } },
      (res) => { let buf = ""; res.on("data", (c) => (buf += c));
                 res.on("end", () => { let j; try { j = JSON.parse(buf || "{}"); } catch { j = { raw: buf }; }
                                       resolve({ status: res.statusCode || 0, body: j }); }); });
    r.on("error", reject); r.on("timeout", () => r.destroy(new Error("vmmanager timeout")));
    if (data) r.write(data); r.end();
  });
}
async function vmHealth(timeoutMs = 3000) {
  const r = await vmReq("GET", "/health", null, timeoutMs);
  if (r.status !== 200) throw new Error(`vmmanager /health ${r.status}`);
  return r.body;
}

// ---- on-chain discovery: self-register in NanRegistry (no trusted gateway) --
// On boot the enclave publishes itself (endpoint + attestation repo) to the
// registry contract on Base, then heartbeats. Callers read the registry from
// any RPC and connect DIRECTLY, verifying attestation themselves. Entirely
// opt-in: if REGISTRY_ENABLED isn't set, the enclave just doesn't advertise.
const REGISTRY_ENABLED  = /^(1|true|on)$/i.test(process.env.REGISTRY_ENABLED || "");
const REGISTRY_ADDRESS  = process.env.REGISTRY_ADDRESS || "";
const REGISTRY_PK       = process.env.REGISTRY_PRIVATE_KEY || "";        // operator key (enclave secret); needs a little Base ETH for gas
const ENCLAVE_ENDPOINT  = (process.env.ENCLAVE_ENDPOINT || PUBLIC_URL || "").replace(/\/+$/, "");
const ENCLAVE_REPO      = process.env.ENCLAVE_REPO || "";                // e.g. "SteveDeFacto/Nan" - what callers attest against
const ENCLAVE_MEASUREMENT = process.env.ENCLAVE_MEASUREMENT || ("0x" + "0".repeat(64)); // optional cross-check
const HEARTBEAT_SEC     = parseInt(process.env.REGISTRY_HEARTBEAT_SEC || "900", 10);
const REGISTRY_ABI = [
  { type: "function", name: "register", stateMutability: "nonpayable",
    inputs: [{ name: "endpoint", type: "string" }, { name: "repo", type: "string" }, { name: "measurement", type: "bytes32" }],
    outputs: [{ name: "id", type: "bytes32" }] },
  { type: "function", name: "heartbeat", stateMutability: "nonpayable",
    inputs: [{ name: "id", type: "bytes32" }], outputs: [] },
];

async function registerOnChain() {
  if (!REGISTRY_ENABLED) return;
  for (const [k, v] of [["REGISTRY_ADDRESS", REGISTRY_ADDRESS], ["REGISTRY_PRIVATE_KEY", REGISTRY_PK],
                        ["ENCLAVE_ENDPOINT", ENCLAVE_ENDPOINT], ["ENCLAVE_REPO", ENCLAVE_REPO]]) {
    if (!v) { console.warn(`[registry] disabled: missing ${k}`); return; }
  }
  try {
    const account = privateKeyToAccount(REGISTRY_PK.startsWith("0x") ? REGISTRY_PK : `0x${REGISTRY_PK}`);
    const wallet  = createWalletClient({ account, chain: base, transport: viemHttp(BASE_RPC) });
    const id = keccak256(stringToBytes(ENCLAVE_ENDPOINT));
    const hash = await wallet.writeContract({
      address: REGISTRY_ADDRESS, abi: REGISTRY_ABI, functionName: "register",
      args: [ENCLAVE_ENDPOINT, ENCLAVE_REPO, ENCLAVE_MEASUREMENT],
    });
    console.log(`[registry] registered ${ENCLAVE_ENDPOINT} repo=${ENCLAVE_REPO} id=${id} tx=${hash}`);
    // heartbeat loop - refresh liveness so readers don't treat us as down
    setInterval(async () => {
      try {
        const h = await wallet.writeContract({ address: REGISTRY_ADDRESS, abi: REGISTRY_ABI,
                                               functionName: "heartbeat", args: [id] });
        console.log(`[registry] heartbeat tx=${h}`);
      } catch (e) { console.warn(`[registry] heartbeat failed: ${e.shortMessage || e.message}`); }
    }, Math.max(60, HEARTBEAT_SEC) * 1000).unref();
  } catch (e) {
    // never fatal - a failed advertisement must not take down the enclave
    console.warn(`[registry] self-registration failed: ${e.shortMessage || e.message}`);
  }
}
// The sandbox sshd host key is GENERATED ONCE AT BOOT inside the enclave and
// measured into a TDX RTMR (see initSshHostKey) - so its fingerprint is
// attestation-bound without baking a key into any image, and one fingerprint
// covers every instance. These are set at runtime, never from env.
let SSH_HOST_KEY_PATH = null;
let SSH_HOST_KEY_FP   = "SHA256:<pending-boot>";

function need(n){ const v = process.env[n]; if(!v){ console.error("FATAL: missing env", n); process.exit(1);} return v; }

// ---- GPU resource model: ARBITRARY splitting ------------------------------
// CC disables MIG, so the card is ONE trust domain and we slice it in SOFTWARE:
// every deployment is a per-tenant worker PROCESS with (a) a VRAM cap and (b) a
// compute (throughput) share. Any split is allowed at GRANULARITY_GB, as long as
// the per-card sums fit. Isolation comes from the process boundary, not the slice
// size (separate GPU address spaces; VRAM scrubbed by the driver on worker exit).
const CPU_RATE        = 0.0000306;                                      // USDC/sec, CPU-only ($0.11/hr)
const FULL_RATE       = 0.0016667;                                      // USDC/sec, a WHOLE card ($6.00/hr)
const GPU_COUNT       = parseInt(process.env.GPU_COUNT || "1", 10);     // cards in this enclave
const CARD_VRAM_GB    = parseFloat(process.env.GPU_VRAM_GB || "141");   // usable VRAM per card
const CTX_OVERHEAD_GB = parseFloat(process.env.CTX_OVERHEAD_GB || "0.5"); // per-worker context cost, reserved on top of the cap
const SM_TOTAL        = parseInt(process.env.SM_TOTAL || "132", 10);   // SMs per card (H200=132); for reporting granted SMs
const MIN_COMPUTE_PCT = parseInt(process.env.MIN_COMPUTE_PCT || "1", 10); // floor; CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is an integer 1..100
const GRANULARITY_GB  = parseFloat(process.env.VRAM_GRANULARITY_GB || "1"); // request rounding; 1 GB ≈ arbitrary

const round1 = (x) => Math.round(x * 10) / 10;
const round3 = (x) => Math.round(x * 1000) / 1000;
const memShareOf = (vramGb) => vramGb / CARD_VRAM_GB;
// compute is dialed by an INTEGER percent (the MPS cap grain) - quantize any
// requested share to whole percent, floored at MIN_COMPUTE_PCT. This is the true
// allocatable unit; there is no finer control and no 1/7 floor.
const quantizePct = (share) => Math.min(100, Math.max(MIN_COMPUTE_PCT, Math.round(share * 100)));

// per-card free pools (vram + compute). With CC on there is exactly one whole
// device per card - no MIG instances to enumerate.
const gpuCards = Array.from({ length: GPU_COUNT }, (_, i) => ({ id: i, uuid: null, vramFree: CARD_VRAM_GB, computeFree: 1 }));

// price = whole-card rate × the LARGER of memory share or compute share.
function rateFor(vramGb, computeShare) {
  if (!(vramGb > 0)) return CPU_RATE;
  return FULL_RATE * Math.max(memShareOf(vramGb), computeShare);
}
// normalize a request: round VRAM up to granularity; quantize compute to the
// integer-percent MPS grain (default it to the memory share if unspecified).
function normalizeReq(vramGb, computeShare) {
  const v = Math.ceil(vramGb / GRANULARITY_GB) * GRANULARITY_GB;
  const raw = (computeShare == null) ? memShareOf(v) : computeShare;
  const pct = quantizePct(raw);                 // whole percent, floored at MIN_COMPUTE_PCT
  return { vramGb: v, computeShare: pct / 100, computePct: pct };
}
// reserve an arbitrary slice on a single card (best-fit on VRAM). Overhead is
// reserved on top of the cap so the sum of live workers never exceeds physical.
function allocGpu(vramGb, computeShare) {
  const needV = vramGb + CTX_OVERHEAD_GB;
  const fit = gpuCards
    .filter(c => c.vramFree >= needV - 1e-9 && c.computeFree >= computeShare - 1e-9)
    .sort((a, b) => (a.vramFree - needV) - (b.vramFree - needV));
  const card = fit[0];
  if (!card) return null;
  card.vramFree -= needV; card.computeFree -= computeShare;
  return { cardId: card.id, vramGb, computeShare, _needV: needV };
}
function releaseGpu(h) {
  if (!h) return;
  const card = gpuCards[h.cardId]; if (!card) return;
  card.vramFree = Math.min(CARD_VRAM_GB, card.vramFree + h._needV);
  card.computeFree = Math.min(1, card.computeFree + h.computeShare);
}
// largest slice a single card can still take (VRAM net of overhead; compute share)
const maxFreeVram    = () => Math.max(0, ...gpuCards.map(c => c.vramFree - CTX_OVERHEAD_GB));
const maxFreeCompute = () => Math.max(0, ...gpuCards.map(c => c.computeFree));

// wait until the Docker Engine socket answers (it may not be ready at boot).
async function waitForDocker(tries = 20, gapMs = 500) {
  for (let i = 0; i < tries; i++) {
    try { const r = await dockerReq("GET", "/version", null, 3000); if (r.status < 400) return true; } catch {}
    await new Promise(r => setTimeout(r, gapMs));
  }
  return false;
}

const _applyGpu = (text) => {
  let got = 0;
  for (const line of text.trim().split("\n")) {
    const [idx, uuid, memMiB] = line.split(",").map(s => s.trim());
    const i = parseInt(idx, 10);
    if (gpuCards[i] && /^GPU-/.test(uuid || "")) {
      gpuCards[i].uuid = uuid; got++;
      const totalGb = parseFloat(memMiB) / 1024;
      if (totalGb > 0) console.log(`[gpu] card ${i} ${uuid} (${totalGb.toFixed(0)}GB)`);
    }
  }
  return got;
};
const GPU_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"];

// Discover card UUIDs (so workers can be pinned). Supervisor image has no
// nvidia-smi, so enumerate via a one-shot CUDA container over the Docker API.
async function discoverGpus() {
  if (/^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) return 0;
  // fast path: local nvidia-smi if present
  try { const { stdout } = await pexec("nvidia-smi", GPU_QUERY.slice(1), { timeout: 8000 });
        const n = _applyGpu(stdout); if (n >= GPU_COUNT) return n; } catch {}
  const ref = process.env.GPU_SCAN_IMAGE || "nvidia/cuda:12.6.2-base-ubuntu24.04";
  const name = WORKER_PREFIX + "gpuscan";
  try {
    await dockerPull(ref);
    await dockerReq("DELETE", `/containers/${name}?force=1`).catch(() => {});
    const created = await dockerJson("POST", `/containers/create?name=${name}`, {
      Image: ref, Cmd: GPU_QUERY,
      HostConfig: { DeviceRequests: [{ Driver: "nvidia", Count: -1, Capabilities: [["gpu"]] }] },
    });
    const cid = created.Id;
    await dockerJson("POST", `/containers/${cid}/start`);
    await dockerReq("POST", `/containers/${cid}/wait`, null, 30000);
    const r = await dockerReq("GET", `/containers/${cid}/logs?stdout=1&stderr=1`);
    await dockerReq("DELETE", `/containers/${cid}?force=1`).catch(() => {});
    return _applyGpu(demuxLogs(r.buf));
  } catch (e) {
    console.warn("[gpu] UUID discovery via docker failed:", e.message);
    return 0;
  }
}

// Lazily ensure UUIDs are known before a GPU spawn - covers a boot-time socket
// race where discovery ran before the Docker socket was ready.
let _gpuDiscovering = null;
async function ensureGpuUuids() {
  if (gpuCards.every(c => c.uuid)) return true;
  if (!_gpuDiscovering) _gpuDiscovering = (async () => { await waitForDocker(); return discoverGpus(); })()
    .finally(() => { _gpuDiscovering = null; });
  await _gpuDiscovering;
  return gpuCards.some(c => c.uuid);
}

async function initGpu() {
  // Best-effort at boot; the socket may not be up yet, so wait briefly. If it
  // still fails, ensureGpuUuids() retries on the first spawn. Never blocks boot.
  await waitForDocker();
  const got = await discoverGpus();
  if (got < GPU_COUNT) console.warn(`[gpu] boot discovery ${got}/${GPU_COUNT} - will retry on first spawn`);
}

async function initMps() {
  // Start the MPS control daemon ONCE at boot. Workers join it as clients (sharing
  // MPS_PIPE_DIR) and the driver enforces, per client, BOTH the SM cap
  // (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) and the VRAM cap (CUDA_MPS_PINNED_DEVICE_MEM_LIMIT)
  // - confirmed enforced under CC via %smid. Without MPS, compute-share is unenforced
  // and we fall back to admission control + watchdog (workers still run).
  if (!ENABLE_MPS) { console.warn("[mps] disabled by env - compute-share will NOT be enforced"); return; }
  try {
    execFileSync("mkdir", ["-p", MPS_PIPE_DIR]);
    // already running? control daemon answers on the pipe dir.
    try { execFileSync("nvidia-cuda-mps-control", ["get_server_list"],
            { env: { ...process.env, CUDA_MPS_PIPE_DIRECTORY: MPS_PIPE_DIR }, stdio: "ignore" });
          console.log("[mps] daemon already running"); return; } catch {}
    execFileSync("nvidia-cuda-mps-control", ["-d"],
      { env: { ...process.env, CUDA_MPS_PIPE_DIRECTORY: MPS_PIPE_DIR } });
    console.log(`[mps] control daemon started (pipe ${MPS_PIPE_DIR})`);
  } catch (e) {
    console.warn("[mps] could not start daemon - compute caps unenforced:", e.message);
  }
}

async function initSshHostKey() {
  // Generate the sandbox sshd host key ONCE, in-enclave. Every per-deployment
  // sshd the supervisor starts uses THIS key, so a single fingerprint covers all
  // instances and is verifiable against attestation. No key is baked into any
  // image. Resilient: if ssh-keygen is absent (local dev), boot continues with a
  // placeholder fingerprint.
  try {
    const dir  = mkdtempSync(join(tmpdir(), "nan-hostkey-"));
    const path = join(dir, "ssh_host_ed25519_key");
    execFileSync("ssh-keygen", ["-t", "ed25519", "-N", "", "-q", "-C", "nan-host", "-f", path]);
    const out  = execFileSync("ssh-keygen", ["-lf", `${path}.pub`]).toString(); // "256 SHA256:… comment (ED25519)"
    SSH_HOST_KEY_PATH = path;
    SSH_HOST_KEY_FP   = (out.match(/SHA256:\S+/) || ["SHA256:<unknown>"])[0];
    // Tinfoil exposes no guest RTMR-extend, so this key cannot be folded into a
    // hardware register; getMeasurements() reports it as measured:false
    // (asserted by attested code) rather than pretending otherwise.
  } catch (e) {
    console.warn("ssh host key not generated (ssh-keygen missing?):", e.message);
  }
}

const chainClient = createPublicClient({ chain: base, transport: viemHttp(BASE_RPC) });

// ----------------------------------------------------------------------------
// state (in-process; this service is the single enclave instance)
// ----------------------------------------------------------------------------
const nonces     = new Map(); // nonce -> { address, exp }
const deployments = new Map(); // id -> record (incl. local container handle)
setInterval(() => { const t = Date.now(); for (const [n,v] of nonces) if (v.exp < t) nonces.delete(n); }, 60_000).unref?.();
const rid = (p) => p + Math.random().toString(36).slice(2, 10);

// ---- state persistence (fair billing across restarts) -----------------------
// Everything billing-critical (deployments, payment cursor, dedup set) is written
// to STATE_FILE so a supervisor restart FREEZES clocks instead of forfeiting them:
// on boot the downtime gap is measured via savedAt and never charged, unpaid
// reservation windows shift by the gap, and the payment watcher resumes scanning
// from the block it last finished - payments made during the outage are credited.
// Without a writable STATE_FILE this degrades to freezing within one process only.
let _stateDirty = false, _statePersistable = true;
function serializeState() {
  const recs = [...deployments.values()].map(r => {
    const o = { ...r };
    for (const k of ["_payTimer", "_respawning"]) delete o[k];   // live handles only
    return o;
  });
  return JSON.stringify({
    savedAt: Date.now(),
    payFromBlock: _payFromBlock == null ? null : _payFromBlock.toString(),
    seenLogs: [..._seenLogs].map(([k, b]) => [k, b.toString()]),
    deployments: recs,
  });
}
function saveStateNow() {
  if (!_statePersistable) return;
  try {
    const tmp = STATE_FILE + ".tmp";
    writeFileSync(tmp, serializeState());
    renameSync(tmp, STATE_FILE);                 // atomic: a crash never leaves a torn file
    _stateDirty = false;
  } catch (e) { console.warn(`[state] save failed: ${e.message}`); }
}
function saveStateSoon() { _stateDirty = true; }
function initStatePersistence() {
  try { mkdirSync(dirname(STATE_FILE), { recursive: true }); }
  catch (e) {
    _statePersistable = false;
    console.warn(`[state] ${STATE_FILE} unavailable (${e.message}) - clocks freeze only within this process`);
    return;
  }
  const t = setInterval(() => { if (_stateDirty) saveStateNow(); }, 2000);
  if (t.unref) t.unref();
  // flush on shutdown so savedAt marks the true start of the outage
  for (const sig of ["SIGTERM", "SIGINT"]) process.on(sig, () => { saveStateNow(); process.exit(0); });
}
function loadState() {
  if (!_statePersistable || !existsSync(STATE_FILE)) return;
  let s; try { s = JSON.parse(readFileSync(STATE_FILE, "utf8")); }
  catch (e) { console.warn(`[state] unreadable (${e.message}) - starting fresh`); return; }
  const gapMs = Math.max(0, Date.now() - (s.savedAt || Date.now()));
  if (s.payFromBlock != null) _payFromBlock = BigInt(s.payFromBlock);   // watcher resumes where it stopped
  for (const [k, b] of s.seenLogs || []) _seenLogs.set(k, BigInt(b));
  let running = 0, waiting = 0;
  for (const r of s.deployments || []) {
    r._payTimer = null; r._respawning = false;
    deployments.set(r.id, r);
    if (r.payRef) payRefIndex.set(r.payRef.toLowerCase(), r.id);
    if (r._gpu) {                       // re-reserve the GPU slice this deployment still holds
      const card = gpuCards[r._gpu.cardId];
      if (card) { card.vramFree -= r._gpu._needV; card.computeFree -= r._gpu.computeShare; }
    }
    if (r.status === "running") {
      // FREEZE the outage: the gap between savedAt and now is never charged. The
      // first healthy tick verifies the instance (respawning it if the backend
      // lost it) and resumes the clock.
      r._lastTickAt = Date.now();
      r.paused = true; r.pauseReason = "restart_recovery";
      r._respawnAt = 0; r._respawnBackoffMs = 0;
      running++;
    } else if (r.status === "awaiting_payment") {
      r.payDeadline = (r.payDeadline || Date.now()) + gapMs;   // reservation window frozen too
      armPayTimer(r);
      waiting++;
    }
  }
  if (deployments.size) console.log(`[state] restored ${deployments.size} deployment(s) `
    + `(${running} running, ${waiting} awaiting payment) after ${Math.round(gapMs / 1000)}s down; clocks were frozen`);
}

// ---- SSH access ------------------------------------------------------------
// Generate an ed25519 keypair in-enclave via ssh-keygen (correct OpenSSH format).
// privateKey is surfaced to the user exactly ONCE, in the create response.
function generateSshKeypair(label) {
  const dir = mkdtempSync(join(tmpdir(), "nan-ssh-"));
  try {
    const key = join(dir, "id");
    execFileSync("ssh-keygen", ["-t", "ed25519", "-N", "", "-q", "-C", label || "nan", "-f", key]);
    return { privateKey: readFileSync(key, "utf8"), publicKey: readFileSync(key + ".pub", "utf8").trim() };
  } finally { rmSync(dir, { recursive: true, force: true }); }
}
// SSH rides the one attested origin over a WebSocket (no extra port): /x/:id/ssh.
function sshCommandFor(endpoint) {
  const wss = endpoint.replace(/^https:/i, "wss:") + "/ssh";
  return `ssh -o ProxyCommand='websocat -b ${wss}' ${SSH_USER}@nan`;
}
// public access shape (NEVER includes the private key)
function sshAccessOf(rec) {
  return { user: SSH_USER, command: sshCommandFor(rec.network.endpoint),
           hostKeyFingerprint: SSH_HOST_KEY_FP, keySource: rec._sshKeySource || "generated" };
}

// ============================================================================
// >>> IMPLEMENT THESE for your CVM launch mechanism (e.g. the app manager on
//     VMMGR_URL). Contract: one ingress port, no sibling reach.
//     Tinfoil exposes no guest RTMR-extend, so a launched image's digest cannot
//     be folded into the hardware measurements; /attestation reports exactly
//     that (per-app `coverage` in getMeasurements) instead of implying it.
//     SSH: the sandbox runs ANY stock image and needs NO sshd of its own. The
//     supervisor hosts sshd (measured host key from initSshHostKey); spawn starts
//     a loopback sshd for this deployment using SSH_HOST_KEY_PATH, installs
//     `authorizedKey`, and sets a ForceCommand that exec's into THIS sandbox's
//     namespace. Return its loopback port as sshPort.
// ============================================================================
// ============================================================================
// WORKER LAUNCH - one container per tenant. The process boundary is the ONLY
// thing giving memory isolation + fault containment + VRAM scrub-on-exit at once
// (all empirically confirmed). Compute + VRAM are capped by MPS, also confirmed
// enforced under CC. Never co-locate two tenants in one process.
//   STILL TODO (separate steps): SSH data-plane (returns sshPort 0 here
//   - the HTTP data path is the real channel; SSH is unwired in this revision).
//   Image digests are NOT RTMR-extended (no guest extend interface) - the
//   attestation endpoint reports that coverage gap explicitly, never fakes it.
// ============================================================================
const containerName = (id) => WORKER_PREFIX + String(id).replace(/[^a-zA-Z0-9_.-]/g, "");
// resolve the pinned image ref: prefer name@sha256:digest when a digest is given
function pinnedRef(image) {
  const ref = (image?.reference || DEFAULT_IMAGE).trim();
  const dig = (image?.digest || "").trim();
  if (ref.includes("@")) return ref;                              // already digest-pinned
  if (/^sha256:[0-9a-f]{64}$/i.test(dig)) return `${ref.replace(/:[^/:]+$/, "")}@${dig}`;
  return ref;                                                     // tag-only (pin verification is the attestation step)
}
function toBytes(s) {
  const m = /^(\d+)\s*([gmk]?)b?$/i.exec(String(s).trim());
  if (!m) return 0;
  const n = +m[1], u = m[2].toLowerCase();
  return u === "g" ? n*1073741824 : u === "m" ? n*1048576 : u === "k" ? n*1024 : n;
}
// docker multiplexed log stream: [stream:1][000][size:4 BE][payload]…  -> plain text
function demuxLogs(buf) {
  let out = "", o = 0;
  while (o + 8 <= buf.length) {
    const size = buf.readUInt32BE(o + 4), start = o + 8, end = start + size;
    if (end > buf.length) break;
    out += buf.slice(start, end).toString(); o = end;
  }
  return o > 0 ? out : buf.toString();   // fallback if the stream wasn't framed
}

// ---- Docker Engine API client (over the mounted unix socket; no docker CLI) --
function dockerReq(method, path, body, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    const data = body != null ? Buffer.from(JSON.stringify(body)) : null;
    const req = http.request({ socketPath: DOCKER_SOCK, method, path,
      headers: { "Content-Type": "application/json", ...(data ? { "Content-Length": data.length } : {}) } },
      (res) => { const chunks = []; res.on("data", c => chunks.push(c));
                 res.on("end", () => resolve({ status: res.statusCode, buf: Buffer.concat(chunks) })); });
    req.on("error", reject);
    req.setTimeout(timeoutMs, () => req.destroy(new Error("docker socket timeout")));
    if (data) req.write(data); req.end();
  });
}
async function dockerJson(method, path, body, timeoutMs) {
  const r = await dockerReq(method, path, body, timeoutMs);
  let j = null; try { j = r.buf.length ? JSON.parse(r.buf.toString()) : null; } catch {}
  if (r.status >= 400) throw new Error(`docker ${method} ${path.split("?")[0]} -> ${r.status} ${j?.message || r.buf.toString().slice(0,200)}`);
  return j;
}
async function dockerPull(ref) {
  let fromImage = ref, tag = "latest";
  const at = ref.indexOf("@");
  if (at >= 0) { fromImage = ref.slice(0, at); tag = ref.slice(at + 1); }      // repo@sha256:…
  else { const c = ref.lastIndexOf(":"), s = ref.lastIndexOf("/"); if (c > s) { fromImage = ref.slice(0, c); tag = ref.slice(c + 1); } }
  const r = await dockerReq("POST", `/images/create?fromImage=${encodeURIComponent(fromImage)}&tag=${encodeURIComponent(tag)}`, null, SPAWN_TIMEOUT_MS);
  if (r.status >= 400) throw new Error(`pull ${ref} -> ${r.status} ${r.buf.toString().slice(0,200)}`);
  // the pull stream returns 200 even on failure; the error rides in the body
  const err = r.buf.toString().split("\n").filter(Boolean)
    .map(l => { try { return JSON.parse(l); } catch { return null; } }).reverse().find(o => o && o.error);
  if (err) throw new Error(`pull ${ref}: ${err.error}`);
}

async function spawnContainer({ deploymentId, gpu, image, share, appPort, ports }) {
  // Two backends. "vm": hand the app reference to the app manager on VMMGR_URL
  // (the wasm-manager runs it as a `wasmtime serve` process). "worker": fork an MPS-capped
  // CUDA child PROCESS (GPU PTX submission). One proportional `share` sets the cap.
  const shareOf = (g) => Math.min(1, Math.max(MIN_COMPUTE_PCT / 100,
                                              g.vramCapGb / CARD_VRAM_GB, g.computeShare));
  if (/^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) {
    console.log(`[mock] ${PROVISION_BACKEND} tenant ${deploymentId}`);
    return { internalPort: 0, sshPort: 0 };
  }

  if (PROVISION_BACKEND === "vm") {
    const ref = image && image.reference;
    if (!ref) throw new Error("VM backend requires an image reference.");
    const s = (share != null) ? share : (gpu ? shareOf(gpu) : 0.1);
    const r = await vmReq("POST", "/vms",
      { image: ref, share: s, appPort: appPort || 8080, name: deploymentId, ports: ports || [] }, SPAWN_TIMEOUT_MS);
    if (r.status !== 201)
      throw new Error(`vmmanager: ${r.body.error || r.body.message || r.status}`);
    console.log(`[spawn-vm] ${deploymentId} image=${ref} share=${s} `
              + `vm=${r.body.id} hostPort=${r.body.hostPort} status=${r.body.status}`);
    // The VM boots asynchronously; the data path 502s until its server is up.
    return { internalPort: r.body.hostPort || 0, sshPort: 0, vmId: r.body.id, hostPort: r.body.hostPort,
             portMap: r.body.portMap || {} };   // logical "tcp:5432" -> actual loopback bind
  }

  // worker backend (GPU)
  if (!gpu) throw new Error("CPU-only deployments are not supported (a GPU share is required).");
  const share2 = shareOf(gpu);
  const r = await mgrReq("POST", "/tenants", { id: deploymentId, share: share2 }, SPAWN_TIMEOUT_MS);
  if (r.status !== 201 || r.body.status !== "running")
    throw new Error(`worker manager: ${r.body.error || r.body.status || r.status} `
                  + `(sm_granted=${r.body.sm_granted ?? "?"})`);
  console.log(`[spawn] tenant=${deploymentId} share=${share2.toFixed(3)} `
            + `sm_granted=${r.body.sm_granted} device=${r.body.device}`);
  return { internalPort: 0, sshPort: 0, smGranted: r.body.sm_granted };
}

async function stopContainer(rec) {
  if (PROVISION_BACKEND === "vm") {
    if (rec._vmId)
      await vmReq("DELETE", `/vms/${encodeURIComponent(rec._vmId)}`)
        .catch((e) => console.warn(`[stop-vm] ${rec.id}: ${e.message}`));
    return;
  }
  // Tear down the tenant's MPS-capped child. The manager terminates the process,
  // which returns its context/VRAM to the driver and releases the share. NOTE:
  // freed VRAM is not zeroed here - residual-data scrubbing is Layer 4.
  await mgrReq("DELETE", `/tenants/${encodeURIComponent(rec.id)}`)
    .catch((e) => console.warn(`[stop] ${rec.id}: ${e.message}`));
}
// What app ran, as an attestation-visible identity. For ipfs://<cid> the CID IS a
// content hash the (attested) wasm-manager verified the bytes against before running,
// so reporting it here is honest: "the enclave ran exactly this CID."
function appMeasurement(rec) {
  const ref = (rec.image && rec.image.reference) || null;
  const m = /^ipfs:\/\/([^/?#]+)/.exec(ref || "");
  return m ? { kind: "ipfs", reference: ref, cid: m[1], verifiedAgainstCid: true,
               coverage: "Bytes were verified against this CID inside the enclave by the attested "
                       + "wasm-manager before launch. The CID itself is NOT in a hardware register." }
           : { kind: "catalog", reference: ref,
               coverage: "Baked into the attested enclave image, so it is covered by the enclave "
                       + "measurement registers below." };
}
// ---- REAL ATTESTATION -------------------------------------------------------
// The Tinfoil shim generates the enclave TLS key, obtains an Intel TDX quote
// whose report_data[0:32] = sha256(TLS pubkey, SPKI DER), and serves the signed
// Remote Attestation Document at /.well-known/tinfoil-attestation. We RELAY that
// document verbatim and PARSE the quote so the registers are inspectable - but
// we never claim "verified": the party being verified cannot vouch for itself.
// Clients verify with github.com/tinfoilsh/verifier against the Sigstore-signed
// measurements published on ENCLAVE_REPO's releases, over their OWN connection.
const RAD_PATH        = "/.well-known/tinfoil-attestation";
const ATTESTATION_URL = process.env.ATTESTATION_URL || "";                 // explicit RAD URL override
const RAD_CACHE_MS    = parseInt(process.env.RAD_CACHE_MS || "300000", 10); // convenience-copy staleness bound
const sha256Hex = (b) => createHash("sha256").update(b).digest("hex");

// GET url, tolerate the shim's cert (the RAD is SELF-verifying: the quote binds
// the TLS key, so transport trust adds nothing), and capture the peer key so we
// can report the fingerprint exactly as tinfoilsh/verifier computes it.
function fetchRad(url, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = (u.protocol === "https:" ? https : http).request(u,
      { method: "GET", timeout: timeoutMs, rejectUnauthorized: false }, (res) => {
        // grab the peer key NOW - the socket detaches from res once the body ends
        let liveTlsKeyFP = null;
        try { const x = res.socket.getPeerX509Certificate?.();
              if (x) liveTlsKeyFP = sha256Hex(x.publicKey.export({ type: "spki", format: "der" })); } catch {}
        let buf = ""; res.on("data", (c) => (buf += c));
        res.on("end", () => {
          if (res.statusCode !== 200) return reject(new Error(`${u.host}${u.pathname}: HTTP ${res.statusCode}`));
          let doc; try { doc = JSON.parse(buf); } catch { return reject(new Error(`${u.host}: not JSON`)); }
          if (typeof doc.format !== "string" || typeof doc.body !== "string")
            return reject(new Error(`${u.host}: not a Tinfoil attestation document`));
          resolve({ doc, liveTlsKeyFP, url });
        });
      });
    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error(`${u.host}: timeout`)));
    req.end();
  });
}
let _radCache = null;                 // { doc, liveTlsKeyFP, url, at }
let _radInflight = null;
async function fetchEnclaveRad(origin) {
  if (_radCache && Date.now() - _radCache.at < RAD_CACHE_MS) return _radCache;
  if (_radInflight) return _radInflight;
  // the shim terminates TLS inside this CVM, so loopback is tried first; the
  // public origin (hairpin through ingress) is the fallback.
  const candidates = ATTESTATION_URL ? [ATTESTATION_URL]
    : ["https://127.0.0.1" + RAD_PATH, "http://127.0.0.1" + RAD_PATH,
       ...(origin ? [origin + RAD_PATH] : [])];
  _radInflight = (async () => {
    let lastErr = null;
    for (const url of candidates) {
      try { const r = await fetchRad(url); _radCache = { ...r, at: Date.now() }; return _radCache; }
      catch (e) { lastErr = e; }
    }
    throw new Error(`attestation document unreachable (${lastErr?.message || "no candidates"})`);
  })();
  try { return await _radInflight; } finally { _radInflight = null; }
}

// Parse a raw Intel TDX quote (DCAP QuoteV4/V5). Offsets are the fixed TD-report
// layout; report_data is what binds the TLS key. Returns null on anything odd -
// the verbatim document is still returned, clients parse it themselves anyway.
function parseTdxQuote(q) {
  try {
    if (q.length < 48 + 584) return null;
    const version = q.readUInt16LE(0), teeType = q.readUInt32LE(4);
    if (teeType !== 0x81) return null;                                  // TDX
    let body;
    if (version === 4) body = q.subarray(48, 48 + 584);
    else if (version === 5) {
      const bodyType = q.readUInt16LE(48);
      if (bodyType !== 2 && bodyType !== 3) return null;                // TD 1.0 / 1.5
      body = q.subarray(54);
    } else return null;
    if (body.length < 584) return null;
    const hx = (o, n) => body.subarray(o, o + n).toString("hex");
    return { quoteVersion: version,
             mrSeam: hx(16, 48), mrTd: hx(136, 48),
             rtmr0: hx(328, 48), rtmr1: hx(376, 48), rtmr2: hx(424, 48), rtmr3: hx(472, 48),
             reportData: hx(520, 64) };
  } catch { return null; }
}
// AMD SEV-SNP report (in case a CVM lands on SNP hardware): fixed offsets too.
function parseSnpReport(r) {
  try {
    if (r.length < 0x90 + 48) return null;
    return { measurement: r.subarray(0x90, 0x90 + 48).toString("hex"),
             reportData:  r.subarray(0x50, 0x50 + 64).toString("hex") };
  } catch { return null; }
}
function parseRad(doc) {
  let raw = Buffer.from(doc.body, "base64");
  if (raw[0] === 0x1f && raw[1] === 0x8b) raw = gunzipSync(raw);         // predicate v2 bodies are gzipped
  const fmt = doc.format || "";
  if (fmt.includes("tdx-guest")) {
    const p = parseTdxQuote(raw);
    return { technology: "intel-tdx", quote: raw.toString("base64"),
             measurements: p && { mrTd: p.mrTd, rtmr0: p.rtmr0, rtmr1: p.rtmr1, rtmr2: p.rtmr2, rtmr3: p.rtmr3 },
             reportData: p?.reportData ?? null, quoteVersion: p?.quoteVersion };
  }
  if (fmt.includes("sev-snp-guest")) {
    const p = parseSnpReport(raw);
    return { technology: "amd-sev-snp", quote: raw.toString("base64"),
             measurements: p && { measurement: p.measurement }, reportData: p?.reportData ?? null };
  }
  return { technology: fmt, quote: raw.toString("base64"), measurements: null, reportData: null };
}

// GPU evidence comes from the worker manager (the container that holds the card):
// NVML's conf-compute attestation report, signed by the GPU, over OUR nonce.
async function fetchGpuEvidence(nonceHex, timeoutMs = 30000) {
  const r = await mgrReq("GET", `/attestation?nonce=${nonceHex}`, null, timeoutMs);
  if (r.status !== 200 || r.body.available === false)
    throw new Error(r.body.error || `worker manager /attestation ${r.status}`);
  return r.body;
}

async function getMeasurements(rec, { origin = ENCLAVE_ENDPOINT, nonce } = {}) {
  const out = {
    // ALWAYS false here: verification is the CLIENT's act, on its own connection.
    // A "verified: true" asserted by the machine being verified proves nothing.
    verified: false,
    verify: {
      how: "Fetch " + RAD_PATH + " from this origin over your OWN TLS connection, verify the quote "
         + "against the Intel/AMD root of trust (github.com/tinfoilsh/verifier does both), compare the "
         + "registers to the Sigstore-signed measurements on the release page of `repo`, and check that "
         + "reportData[0:32] equals sha256 of the TLS public key (SPKI DER) your connection presents.",
      repo: ENCLAVE_REPO || null,
      attestationEndpoint: (origin || "") + RAD_PATH,
    },
    tlsKeyFingerprint: null,
    sshHostKeyFingerprint: SSH_HOST_KEY_FP,
    sshHostKey: { fingerprint: SSH_HOST_KEY_FP, measured: false,
                  note: "Generated at boot inside the enclave by the attested supervisor and served over "
                      + "the attested origin, but NOT folded into a hardware register (Tinfoil exposes no "
                      + "guest RTMR-extend), so it is asserted by measured code rather than measured itself." },
    app: rec ? appMeasurement(rec) : null,
    vm: null,
    gpu: null,
  };
  try {
    const { doc, liveTlsKeyFP, url } = await fetchEnclaveRad(origin);
    const parsed = parseRad(doc);
    const attestedTlsFP = parsed.reportData ? parsed.reportData.slice(0, 64) : null;
    out.tlsKeyFingerprint = attestedTlsFP ? `sha256:${attestedTlsFP}` : null;
    out.enclave = {
      attestationDocument: doc,               // verbatim Tinfoil RAD - feed to tinfoilsh/verifier
      fetchedFrom: url, fetchedAt: new Date(_radCache.at).toISOString(),
      // fingerprint of the key the shim ACTUALLY presented when we fetched; equals
      // the quote-bound one unless the shim rotated its key mid-cache-window.
      observedTlsKeyFingerprint: liveTlsKeyFP,
    };
    out.vm = { technology: parsed.technology, quote: parsed.quote, quoteVersion: parsed.quoteVersion,
               measurements: parsed.measurements, reportData: parsed.reportData };
  } catch (e) {
    out.enclave = { available: false, error: e.message,
                    note: "Fetch " + RAD_PATH + " from this origin yourself - the shim serves it directly." };
  }
  if (rec?._gpu) {
    const n = nonce || randomBytes(32).toString("hex");
    try {
      const ev = await fetchGpuEvidence(n);
      out.gpu = { technology: "nvidia-cc", ccMode: ev.ccMode ?? null, nonce: ev.nonce || n,
                  driverVersion: ev.driverVersion ?? null,
                  // first card's material at the top level (single-card enclaves); all cards in gpus[]
                  report: ev.gpus?.[0]?.attestationReport_b64 ?? null,
                  certChain: ev.gpus?.[0]?.attestationCertChain_b64 ?? null,
                  gpus: ev.gpus || [],
                  vramCapGb: rec.resources.vramGb, computeShare: rec.resources.computeShare,
                  verify: "Check the report + cert chain with NVIDIA NRAS or nvtrust's local_gpu_verifier; "
                        + "confirm it signs YOUR nonce. The whole card is one CC trust domain - the VRAM/compute "
                        + "split is enforced by the attested supervisor+MPS, not by the hardware report." };
    } catch (e) {
      out.gpu = { technology: "nvidia-cc", available: false, error: e.message };
    }
  }
  return out;
}
function capacity() {
  return {
    cpu: 64,
    gpu: gpuCards.map(c => ({ id: c.id, vramFreeGb: round1(c.vramFree),
                              computeFree: round3(c.computeFree),
                              computeFreePct: Math.round(c.computeFree * 100),
                              smFree: Math.round(c.computeFree * SM_TOTAL) })),
    vramFreeGb: round1(gpuCards.reduce((s, c) => s + c.vramFree, 0)),
    maxVramGb: round1(maxFreeVram()),
    maxComputeShare: round3(maxFreeCompute()),
  };
}
// ============================================================================

const app = express();
app.use(cors({
  origin: CORS_ORIGINS.includes("*") ? true : CORS_ORIGINS,
  methods: ["GET","POST","PATCH","DELETE","OPTIONS"],
  allowedHeaders: ["Authorization","Content-Type"],
  maxAge: 86400,
}));

const fail = (res, status, code, message) => res.status(status).json({ code, message });
const originOf = (req) => PUBLIC_URL || `https://${req.headers["x-forwarded-host"] || req.headers.host}`;

async function addrFromAuth(req) {
  const m = (req.headers.authorization || "").match(/^Bearer\s+(.+)$/i);
  if (!m) return null;
  try { const { payload } = await jwtVerify(m[1], SECRET); return getAddress(payload.sub); }
  catch { return null; }
}

// ---------------------------------------------------------------------------
// DATA PATH - registered BEFORE express.json() so the body streams untouched.
// Same token, same origin as control; supervisor checks ownership, then proxies.
// ---------------------------------------------------------------------------
// Firewall config validation. Mirrors the wasm-manager's rules so a bad spec fails
// fast at create (422) instead of at provision: entries are "http" (default serve
// mode) | "http:N" | "tcp:N" | "udp:N"; N in 1..19999 (logical labels; <1024 always remapped), excluding infra ports
// (8080 supervisor, 8091 manager) and the manager-assigned serve range (20000+).
const FW_MIN = 1, FW_MAX = 19999, FW_RESERVED = new Set([8080, 8091]);   // logical labels; <1024 is always remapped to an unprivileged actual by the manager
function parseFirewall(fw) {
  const raw = (fw && Array.isArray(fw.ports)) ? fw.ports : [];
  if (raw.length > 8) throw new Error("firewall.ports: at most 8 entries.");
  const out = [];
  for (const e of raw) {
    const s = String(e).trim().toLowerCase();
    if (!s || s === "http") continue;                       // default serve mode marker
    const m = /^(http|tcp|udp):(\d{1,5})$/.exec(s);
    if (!m) throw new Error(`firewall.ports: bad entry "${e}" (use http[:N] | tcp:N | udp:N).`);
    const p = +m[2];
    if (p < FW_MIN || p > FW_MAX || FW_RESERVED.has(p))
      throw new Error(`firewall.ports: port ${p} not allowed (${FW_MIN}-${FW_MAX}, excluding ${[...FW_RESERVED].join("/")}).`);
    if (!out.includes(m[1] + ":" + p)) out.push(m[1] + ":" + p);
  }
  if (out.filter((x) => x.startsWith("http:")).length > 1) throw new Error("firewall.ports: only one http:N entry.");
  return out;                                               // [] = classic wasi:http serve mode
}
const fwTcpPorts = (rec) => (rec.firewall || []).filter((x) => x.startsWith("tcp:")).map((x) => +x.slice(4));

app.use("/x/:id", async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec) return fail(res, 404, "not_found", "Unknown deployment.");
  // Public deployments serve anyone (websites/APIs). Private ones require the owner's
  // token (checked before status so a private deployment's state isn't leaked). SSH
  // (the WebSocket upgrade below) is ALWAYS owner-only, regardless of `public`.
  if (!rec.public) {
    const addr = await addrFromAuth(req);
    if (!addr) return fail(res, 401, "unauthorized", "Missing or invalid token.");
    if (rec.owner !== addr) return fail(res, 403, "forbidden", "Not your deployment.");
  }
  if (rec.status !== "running") return fail(res, 409, "not_running", `Deployment is ${rec.status}.`);

  // vm backend: proxy to the app's loopback port on the app manager, on shared
  // localhost. worker backend: /x/:id/<sub> -> /tenants/:id/<sub>.
  const sub = req.url.replace(/^\/+/, "");
  let target;
  if (PROVISION_BACKEND === "vm") {
    if (!rec._vmHostPort) {
      return (rec.firewall && rec.firewall.length && !rec.firewall.some((x) => x.startsWith("http")))
        ? fail(res, 502, "no_http", "This app exposes raw TCP/UDP ports, not HTTP. Reach declared TCP ports via the WebSocket bridge at /x/:id/tcp/:port (e.g. websocat).")
        : fail(res, 502, "vm_not_ready", "The VM has no forwarded port yet.");
    }
    target = new URL(`http://127.0.0.1:${rec._vmHostPort}/${sub}`);
  } else {
    target = new URL(`${WORKER_MGR_URL}/tenants/${encodeURIComponent(req.params.id)}/${sub}`);
  }
  const headers = { ...req.headers, host: target.host };
  delete headers.authorization; // the NAN token stays at the supervisor; the worker never sees it
  const up = http.request(
    { host: target.hostname, port: target.port || 80, method: req.method,
      path: target.pathname + target.search, headers },
    (upRes) => { res.writeHead(upRes.statusCode || 502, upRes.headers); upRes.pipe(res); });
  up.on("error", (e) => { if (!res.headersSent) res.writeHead(502); res.end("upstream error: " + e.message); });
  req.pipe(up);
});

app.use(express.json({ limit: "256kb" }));

async function authed(req, res, next) {
  const addr = await addrFromAuth(req);
  if (!addr) return fail(res, 401, "unauthorized", "Missing or invalid session.");
  req.address = addr; next();
}

// ============================================================================
// system
// ============================================================================
app.get("/v1/health", (_req, res) => res.json({ status: "ok", deployments: deployments.size,
  // watcher freshness is billing-critical: while it's stale, funded clocks are frozen
  watcher: FORWARDER_ADDRESS ? { lastPollOkAt: _lastPollOkAt ? new Date(_lastPollOkAt).toISOString() : null,
                                 fresh: (Date.now() - _lastPollOkAt) < WATCHER_STALE_SEC * 1000 } : null }));
app.get("/v1/version", (_req, res) => res.json({ service: "nan-supervisor/0.1.0", contract: "nan-openapi/1.0.0", chainId: CHAIN_ID }));

app.get("/v1/pricing", (_req, res) => res.json({
  assets: ["ETH","USDC"],
  model: "Request any VRAM slice (GB) and an optional compute share of a card. Both are hardware-enforced caps via MPS under NVIDIA CC (CC disables MIG, so splits are arbitrary, not fixed profiles). Compute is dialed in whole percent of the card. Billed per second by the LARGER of memory share (vramGb / cardVramGb) or compute share, times the whole-card rate.",
  card: { vramGb: CARD_VRAM_GB, count: GPU_COUNT, sms: SM_TOTAL,
          wholeCardPerSecondUsdc: FULL_RATE.toFixed(7), wholeCardPerHourUsdc: (FULL_RATE * 3600).toFixed(2) },
  cpu: { ratePerSecondUsdc: CPU_RATE.toFixed(7), ratePerHourUsdc: (CPU_RATE * 3600).toFixed(2) },
  computeGranularity: { unit: "percent", step: 1, minPercent: MIN_COMPUTE_PCT, note: `compute is set by MPS in whole-percent steps (~${round1(SM_TOTAL/100)} SMs each)` },
  vramGranularityGb: GRANULARITY_GB,
  formula: "ratePerSecondUsdc = wholeCardPerSecond × max(vramGb / cardVramGb, computeShare)",
  examples: [18, 35, 70, 141].map(v => {
    const s = normalizeReq(v, null), r = rateFor(s.vramGb, s.computeShare);
    return { vramGb: s.vramGb, computeShare: round3(s.computeShare),
             billedOn: memShareOf(s.vramGb) >= s.computeShare ? "memory" : "compute",
             ratePerSecondUsdc: r.toFixed(7), ratePerHourUsdc: (r * 3600).toFixed(2) };
  }),
  billingIncrementSeconds: 1,
}));

app.get("/availability", async (_req, res) => {
  // Live from the active backend's allocator. In vm mode the manager reports CPU/RAM
  // capacity; we keep emitting the GPU-flavored fields (synthesized from maxShare) so
  // the existing frontend stays consistent. Falls back to local accounting if down.
  try {
    const h = PROVISION_BACKEND === "vm" ? await vmHealth() : await mgrHealth();
    const c = h.capacity;
    const smFree = (c.smFree != null) ? c.smFree : Math.round(c.maxShare * SM_TOTAL);
    const vramFreeGb = (c.vramFreeGb != null) ? c.vramFreeGb : round1(c.maxShare * CARD_VRAM_GB);
    return res.json({
      maxShare: c.maxShare, usedShare: c.usedShare,
      smFree, smTotal: SM_TOTAL,
      vramFreeGb, cardVramGb: CARD_VRAM_GB, cards: GPU_COUNT,
      source: PROVISION_BACKEND === "vm" ? "vmmanager" : "worker", updatedAt: new Date().toISOString(),
    });
  } catch (e) {
    const c = capacity();
    return res.json({
      maxShare: c.maxComputeShare, usedShare: round3(1 - c.maxComputeShare),
      smFree: Math.round(c.maxComputeShare * SM_TOTAL), smTotal: SM_TOTAL,
      vramFreeGb: c.vramFreeGb, cardVramGb: CARD_VRAM_GB, cards: GPU_COUNT,
      source: "fallback", note: "worker manager unreachable", updatedAt: new Date().toISOString(),
    });
  }
});

// External proof that MPS caps are live: each running tenant's granted SM count
// (sanitized - no tenant ids). A 25% tenant should report ~33 of 132 SMs.
app.get("/v1/gpu", async (_req, res) => {
  try {
    const h = await mgrHealth(5000);
    res.json({
      ok: true, role: h.role, mpsActive: !!h.mps_pipe, capacity: h.capacity, smTotal: SM_TOTAL,
      tenants: (h.tenants || []).map((t) => ({ pct: t.pct, status: t.status, smGranted: t.sm_granted })),
    });
  } catch (e) {
    res.status(503).json({ ok: false, error: `worker manager unreachable: ${e.message}` });
  }
});

// ============================================================================
// auth (SIWE)
// ============================================================================
app.get("/v1/auth/nonce", (req, res) => {
  let address; try { address = getAddress(String(req.query.address || "")); }
  catch { return fail(res, 422, "invalid_address", "Provide a valid ?address."); }
  const nonce = rid("");
  const issuedAt = new Date(), expirationTime = new Date(issuedAt.getTime() + 10 * 60_000);
  nonces.set(nonce, { address, exp: expirationTime.getTime() });
  const statement = "Sign in to NAN. This signature is free and will not move funds.";
  const message =
    `${SIWE_DOMAIN} wants you to sign in with your Ethereum account:\n${address}\n\n${statement}\n\n` +
    `URI: ${SIWE_URI}\nVersion: 1\nChain ID: ${CHAIN_ID}\nNonce: ${nonce}\n` +
    `Issued At: ${issuedAt.toISOString()}\nExpiration Time: ${expirationTime.toISOString()}`;
  res.json({ address, message, nonce, statement, domain: SIWE_DOMAIN, uri: SIWE_URI, version: "1",
             chainId: CHAIN_ID, issuedAt: issuedAt.toISOString(), expirationTime: expirationTime.toISOString() });
});

app.post("/v1/auth/login", async (req, res) => {
  const { message, signature } = req.body || {};
  if (!message || !signature) return fail(res, 422, "invalid_request", "message and signature are required.");
  const nm = message.match(/\nNonce: (\S+)\n/), am = message.match(/^(0x[0-9a-fA-F]{40})$/m);
  if (!nm || !am) return fail(res, 422, "invalid_message", "Malformed SIWE message.");
  const nonce = nm[1], claimed = getAddress(am[1]), rec = nonces.get(nonce);
  if (!rec || rec.exp < Date.now()) { nonces.delete(nonce); return fail(res, 401, "bad_nonce", "Unknown or expired nonce."); }
  if (getAddress(rec.address) !== claimed) return fail(res, 401, "address_mismatch", "Address does not match nonce.");
  let ok = false; try { ok = await verifyMessage({ address: claimed, message, signature }); } catch {}
  if (!ok) return fail(res, 401, "bad_signature", "Signature verification failed.");
  nonces.delete(nonce);
  const expiresAt = new Date(Date.now() + SESSION_TTL * 1000);
  const token = await new SignJWT({}).setProtectedHeader({ alg: "HS256" }).setSubject(claimed)
    .setExpirationTime(expiresAt.getTime() / 1000 | 0).sign(SECRET);
  res.json({ token, tokenType: "Bearer", address: claimed, expiresAt: expiresAt.toISOString() });
});

// ============================================================================
// payments (pay-per-deploy) - the supervisor WATCHES the NanPay forwarder on
// Base for Paid events and converts each payment into runtime. No held balance.
// (outbound Base RPC required - confirm the CVM egress allows BASE_RPC.)
// ============================================================================
const PAY_EVENT = { type: "event", name: "Paid", inputs: [
  { name: "deploymentId", type: "bytes32", indexed: true },
  { name: "payer",        type: "address", indexed: true },
  { name: "amount",       type: "uint256", indexed: false } ] };
const PAY_ETH_EVENT = { type: "event", name: "PaidEth", inputs: [
  { name: "deploymentId", type: "bytes32", indexed: true },
  { name: "payer",        type: "address", indexed: true },
  { name: "amountWei",    type: "uint256", indexed: false } ] };

const payRefIndex = new Map();   // payRef (hex, lowercase) -> deployment id

// USDC (6dp) funded at `rate` USDC/sec buys this many seconds of runtime.
const usdcToSeconds = (amountRaw, rate) => (Number(amountRaw) / 1e6) / (rate || 1);

// --- ETH payments: priced via the Chainlink ETH/USD feed on Base -------------
// Feed address verified on-chain (description() == "ETH / USD", decimals() == 8).
// On another chain (e.g. Base Sepolia) set ETH_USD_FEED to that chain's feed.
const ETH_USD_FEED = process.env.ETH_USD_FEED || "0x71041dddad3595F9CEd3DcCFBe3D1F4b0a16Bb70";
const ETH_FEED_MAX_AGE_SEC = parseInt(process.env.ETH_FEED_MAX_AGE_SEC || "21600", 10); // 6h (feed heartbeat ~20min)
const FEED_ABI = [{ type: "function", name: "latestRoundData", stateMutability: "view", inputs: [],
  outputs: [{ type: "uint80" }, { type: "int256" }, { type: "uint256" }, { type: "uint256" }, { type: "uint80" }] }];

// wei (1e18) * price (8dp) -> USDC-equivalent (6dp):  / 1e20
const weiToUsd6 = (wei, price8) => (wei * price8) / 10n ** 20n;

let _ethUsd = { price8: null, at: 0 };            // cached oracle read (for instructions + conversion)
async function ethUsdPrice8() {
  if (_ethUsd.price8 && (Date.now() - _ethUsd.at) < 60_000) return _ethUsd.price8;
  const [, answer, , updatedAt] = await chainClient.readContract({
    address: getAddress(ETH_USD_FEED), abi: FEED_ABI, functionName: "latestRoundData" });
  const ageSec = Math.floor(Date.now() / 1000) - Number(updatedAt);
  if (answer <= 0n) throw new Error(`ETH/USD feed returned ${answer}`);
  if (ageSec > ETH_FEED_MAX_AGE_SEC) throw new Error(`ETH/USD feed stale (${ageSec}s old)`);
  _ethUsd = { price8: answer, at: Date.now() };
  return answer;
}

// ETH payments retry if the oracle read fails: a payment must never be lost to a
// flaky RPC. Queue drains at the top of every poll tick.
const _pendingEth = [];
async function onPaidEth(payRefHex, payer, wei) {
  try {
    const price8 = await ethUsdPrice8();
    const usd6 = weiToUsd6(wei, price8);
    console.log(`[pay] eth ${wei} wei @ $${(Number(price8) / 1e8).toFixed(2)} -> ${(Number(usd6) / 1e6).toFixed(2)} USDC-equiv (${payRefHex})`);
    await onPaid(payRefHex, payer, usd6);
  } catch (e) {
    _pendingEth.push({ payRefHex, payer, wei });
    console.warn(`[pay] eth payment queued for retry (${e.shortMessage || e.message})`);
  }
}

// --- USDC EIP-712 domain: payers sign an EIP-3009 ReceiveWithAuthorization ---
// against the TOKEN's own domain, so instructions must carry its exact fields.
// name()/version() differ per deployment (mainnet USDC: "USD Coin"; Base Sepolia
// testnet USDC: "USDC"), so read them from the token once and cache forever.
const ERC20_META_ABI = [
  { type: "function", name: "name",    stateMutability: "view", inputs: [], outputs: [{ type: "string" }] },
  { type: "function", name: "version", stateMutability: "view", inputs: [], outputs: [{ type: "string" }] },
];
let _usdcDomain = null;
async function refreshUsdcDomain() {
  if (_usdcDomain) return _usdcDomain;
  const addr = getAddress(USDC_ADDRESS);
  const [name, version] = await Promise.all([
    chainClient.readContract({ address: addr, abi: ERC20_META_ABI, functionName: "name" }),
    chainClient.readContract({ address: addr, abi: ERC20_META_ABI, functionName: "version" }).catch(() => "2"), // FiatTokenV2+ is "2"
  ]);
  _usdcDomain = { name, version, chainId: CHAIN_ID, verifyingContract: addr };
  console.log(`[pay] USDC EIP-712 domain: name="${name}" version="${version}"`);
  return _usdcDomain;
}

function paymentInstructions(rec) {
  return {
    chainId: CHAIN_ID, asset: "USDC", assets: ["USDC", "ETH"], usdc: USDC_ADDRESS,
    forwarder: FORWARDER_ADDRESS || null,
    deploymentRef: rec.payRef,                       // bytes32 to pass to payWithAuthorization() / payEth()
    ratePerSecondUsdc: (rec.rate || 0).toFixed(7),
    method: "payWithAuthorization(bytes32 deploymentId, address from, uint256 value, uint256 validAfter, uint256 validBefore, bytes32 nonce, bytes signature)",
    payEthMethod: "payEth(bytes32 deploymentId) payable",
    usdcDomain: _usdcDomain,                         // EIP-712 domain to sign ReceiveWithAuthorization against; null until the first token read
    ethUsd: _ethUsd.price8 ? (Number(_ethUsd.price8) / 1e8).toFixed(2) : null,   // cached Chainlink read; null until first refresh
    note: "USDC (EIP-3009, no approve): sign a USDC ReceiveWithAuthorization (EIP-712, to = forwarder, "
        + "nonce = first 16 bytes of deploymentRef + 16 random bytes), then anyone submits payWithAuthorization; "
        + "amount(6dp)/rate = seconds. "
        + "ETH: payEth(deploymentRef) with msg.value; credited as USDC-equivalent at the live Chainlink ETH/USD rate.",
  };
}

app.get("/v1/account", authed, (req, res) => {
  const mine = [...deployments.values()].filter(d => d.owner === req.address);
  res.json({
    address: req.address, chainId: CHAIN_ID,
    payment: { forwarder: FORWARDER_ADDRESS || null, usdc: USDC_ADDRESS, asset: "USDC", assets: ["USDC", "ETH"] },
    deployments: {
      running: mine.filter(d => d.status === "running").length,
      awaitingPayment: mine.filter(d => d.status === "awaiting_payment").length,
      total: mine.length,
      totalTimeRemainingSec: mine.reduce((s, d) => s + (timeRemainingSec(d) || 0), 0),
    },
  });
});

// ============================================================================
// deployments
// ============================================================================
// remainingMs === null means unlimited (auto-provision pilot); otherwise it only
// drains on healthy billing ticks, so it IS the truth even mid-outage.
const timeRemainingSec = (rec) => rec.remainingMs == null ? null : Math.max(0, Math.round(rec.remainingMs / 1000));
const spentOf = (rec) => (((rec.consumedMs || 0) / 1000) * (rec.rate || 0)).toFixed(2);
const view = (rec) => {
  const o = { ...rec };
  for (const k of ["_port", "_gpu", "_gpuSpec", "rate", "_sshPort", "_sshKeySource", "_authorizedKey", "_payTimer",
                   "remainingMs", "consumedMs", "_lastTickAt", "_respawnAt", "_respawnBackoffMs", "_respawning"]) delete o[k];
  o.ssh = sshAccessOf(rec);
  o.ratePerSecondUsdc = (rec.rate || 0).toFixed(7);
  o.spentUsdc = spentOf(rec);
  o.paidUsdc = ((rec.paidUsdc || 0) / 1e6).toFixed(2);
  o.timeRemainingSec = timeRemainingSec(rec);
  // an ESTIMATE only: the balance drains solely while service is healthy, so a
  // frozen (paused) deployment has no meaningful wall-clock expiry.
  o.expiresAt = (rec.remainingMs != null && rec.status === "running" && !rec.paused)
    ? new Date(Date.now() + Math.max(0, rec.remainingMs)).toISOString() : null;
  o.payment = paymentInstructions(rec);
  return o;
};

// Arm (or re-arm after a restart) the unpaid-reservation timer from payDeadline.
// If the payment watcher is blind when the deadline hits, the reservation is
// frozen too: a payment may already sit in the unscanned window, so expiry
// defers until the watcher has caught back up to the chain tip.
function armPayTimer(rec) {
  if (rec._payTimer) clearTimeout(rec._payTimer);
  rec._payTimer = setTimeout(() => {
    if (rec.status !== "awaiting_payment") return;
    if (FORWARDER_ADDRESS && (Date.now() - _lastPollOkAt) > WATCHER_STALE_SEC * 1000) {
      rec.payDeadline = Date.now() + 60_000;
      armPayTimer(rec); saveStateSoon();
      return;
    }
    if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
    rec.status = "expired"; rec.error = "unpaid";
    console.log(`[pay] ${rec.id} reservation released (unpaid after payment window)`);
    saveStateSoon();
  }, Math.max(0, rec.payDeadline - Date.now()));
  if (rec._payTimer.unref) rec._payTimer.unref();
}

// ---- app approval (vm backend): catalog-gated deploys -----------------------
// One eth_call resolves a CID to its listing + deployability flags.
const CATALOG_ABI = [{ type: "function", name: "cidStatus", stateMutability: "view",
  inputs: [{ name: "cid", type: "string" }],
  outputs: [
    { name: "listed",    type: "bool"    },
    { name: "appId",     type: "bytes32" },
    { name: "index",     type: "uint256" },
    { name: "approval",  type: "uint8"   },   // 0 pending | 1 approved | 2 rejected
    { name: "yanked",    type: "bool"    },
    { name: "appActive", type: "bool"    },
  ] }];
const BARE_CID_RE = /^(baf[a-z0-9]{10,}|Qm[1-9A-HJ-NP-Za-km-z]{20,})$/;
// Baked-in app ids only — no paths or .wasm filenames. The wasm-manager also
// resolves bare paths under its APPS_DIR (e.g. a cached ipfs-<cid>.wasm), which
// would dodge the approval check, so those never get past this API.
const BAKED_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

// Gate a vm-backend app reference on catalog approval. Returns { ref } (the
// reference to run, bare CIDs normalized to ipfs://) or { error }. An RPC
// failure REJECTS the deploy (fail closed): this is the enforcement point, so
// an outage must not waive it.
async function gateAppReference(reference) {
  const deny = (status, code, msg) => ({ error: { status, code, msg } });
  let ref = String(reference || "").trim();
  if (BARE_CID_RE.test(ref)) ref = "ipfs://" + ref;
  const m = /^ipfs:\/\/([^/?#]+)/.exec(ref);
  if (!m) {
    if (BAKED_ID_RE.test(ref)) return { ref };            // baked-in id: part of the attested image
    return deny(422, "invalid_spec", "image.reference must be a baked-in app id or an ipfs://<cid> listed in the app catalog.");
  }
  if (!APP_CATALOG_ADDRESS)
    return deny(503, "approval_unavailable", "Catalog apps are disabled on this enclave: APP_CATALOG_ADDRESS is not configured, so approval cannot be verified.");
  let st;
  try {
    st = await chainClient.readContract({ address: getAddress(APP_CATALOG_ADDRESS),
      abi: CATALOG_ABI, functionName: "cidStatus", args: [m[1]] });
  } catch (e) {
    console.warn(`[approval] cidStatus(${m[1]}) failed: ${e.shortMessage || e.message}`);
    return deny(503, "catalog_unreachable", "Could not verify this app's approval against the on-chain catalog; try again shortly.");
  }
  const [listed, , , approval, yanked, appActive] = st;
  if (!listed)               return deny(403, "not_approved", "This CID is not listed in the app catalog. Publish it, then ask the catalog owner to approve it.");
  if (!appActive)            return deny(403, "not_approved", "This app is delisted from the catalog.");
  if (yanked)                return deny(403, "not_approved", "This version was yanked by its publisher.");
  if (Number(approval) === 2) return deny(403, "not_approved", "This version was rejected by the catalog owner.");
  if (Number(approval) !== 1) return deny(403, "not_approved", "This version is awaiting catalog-owner approval; it cannot be deployed yet.");
  return { ref };
}

app.post("/v1/deployments", authed, async (req, res) => {
  const b = req.body || {};
  let image = (b.image && b.image.reference) ? b.image : { reference: DEFAULT_IMAGE };
  // Approval gate (vm backend runs catalog apps): only baked-in ids and ipfs://
  // CIDs whose version the catalog owner APPROVED may deploy. Checked before any
  // reservation so a refused app never holds capacity or a payment window.
  if (PROVISION_BACKEND === "vm") {
    const g = await gateAppReference(image.reference);
    if (g.error) return fail(res, g.error.status, g.error.code, g.error.msg);
    image = { ...image, reference: g.ref };
  }
  const appPort = Number(b.port) || 8080;
  // Public endpoint: anyone can reach the app's data path (hosting a website/API).
  // Private (default): only the owner's SIWE token can. SSH/management stay owner-only
  // either way. Confidentiality is unchanged — the TEE still hides the app from the
  // operator; "public" only governs who may send it requests.
  const isPublic = b.public === true || b.public === "true";
  // Firewall: the app's per-version ports config from the catalog ("http" | "http:N"
  // | "tcp:N" | "udp:N"). The wasm-manager grants wasi:sockets and enforces that the
  // app binds ONLY these (bind audit kills violators). Declared TCP ports are reached
  // through the one attested origin as a WebSocket bridge at /x/:id/tcp/:port.
  let firewall;
  try { firewall = parseFirewall(b.firewall); }
  catch (e) { return fail(res, 422, "invalid_spec", e.message); }
  if (b.sshPublicKey != null && !/^(ssh-ed25519|ssh-rsa|ecdsa-sha2-|sk-ssh-|sk-ecdsa-)/.test(String(b.sshPublicKey).trim()))
    return fail(res, 422, "invalid_spec", "sshPublicKey must be an OpenSSH public key (ssh-ed25519 / ssh-rsa / ecdsa / sk-*).");

  // resource request: single `share` (0..1) preferred; legacy vramGb/computeShare still accepted.
  let vramGb0, computeShare0;
  if (b.resources && b.resources.share != null) {
    const s = Number(b.resources.share);
    if (!(s > 0 && s <= 1)) return fail(res, 422, "invalid_spec", "resources.share must be in (0, 1].");
    vramGb0 = s * CARD_VRAM_GB; computeShare0 = s;
  } else {
    vramGb0 = Number((b.resources && b.resources.vramGb) || 0);
    if (!(vramGb0 >= 0) || vramGb0 > 100000) return fail(res, 422, "invalid_spec", "resources.vramGb out of range.");
    computeShare0 = (b.resources && b.resources.computeShare != null) ? Number(b.resources.computeShare) : null;
    if (computeShare0 != null && !(computeShare0 > 0 && computeShare0 <= 1))
      return fail(res, 422, "invalid_spec", "resources.computeShare must be in (0, 1].");
  }
  if (!(vramGb0 > 0)) return fail(res, 422, "invalid_spec", "a GPU share is required (CPU-only is not supported).");

  // reserve an arbitrary GPU slice; the worker isn't spawned until payment lands
  const slice = normalizeReq(vramGb0, computeShare0);
  if (slice.vramGb > maxFreeVram() + 1e-9)
    return fail(res, 422, "invalid_spec", `requested ${slice.vramGb}GB VRAM exceeds the largest free slice (${round1(maxFreeVram())}GB on a ${CARD_VRAM_GB}GB card).`);
  const gpu = allocGpu(slice.vramGb, slice.computeShare);
  if (!gpu) return fail(res, 409, "no_capacity",
    `No single card has ${slice.vramGb}GB VRAM and ${round3(slice.computeShare)} compute share free together (max free: ${round1(maxFreeVram())}GB, ${round3(maxFreeCompute())} share).`);
  const rate = rateFor(slice.vramGb, slice.computeShare);

  // SSH: install the caller's key, or mint one in-enclave and return it ONCE (now, at create).
  let keySource = "provided", authorizedKey = (b.sshPublicKey || "").trim(), oneTimePrivateKey = null;
  if (!authorizedKey) {
    try { const kp = generateSshKeypair(`nan:${req.address.slice(0, 10)}`);
          authorizedKey = kp.publicKey; oneTimePrivateKey = kp.privateKey; keySource = "generated"; }
    catch (e) { releaseGpu(gpu); return fail(res, 500, "keygen_error", "Could not generate an SSH key: " + e.message); }
  }

  const id = rid("dep_");
  const payRef = keccak256(stringToBytes(id));          // the bytes32 to pass to NanPay.payWithAuthorization()
  const rec = {
    id, owner: req.address, status: "awaiting_payment", public: isPublic, firewall,
    image, command: b.command || [],
    resources: { vramGb: slice.vramGb, computeShare: round3(slice.computeShare), share: round3(slice.computeShare), cardId: gpu.cardId },
    network: { port: appPort, protocol: "https", endpoint: `${originOf(req)}/x/${id}` },
    attestation: { available: true, vmTechnology: "intel-tdx", gpuTechnology: "nvidia-cc", href: `/v1/deployments/${id}/attestation` },
    region: "tinfoil", createdAt: new Date().toISOString(), startedAt: null,
    // fair-billing clock: a funded BALANCE (null = unlimited pilot) drained only
    // on healthy ticks - see startBillingTicker. paused surfaces a frozen clock.
    remainingMs: null, consumedMs: 0, paused: false, pauseReason: null, _lastTickAt: 0,
    payDeadline: Date.now() + PAYMENT_WINDOW_SEC * 1000,
    digest: image.digest || null, rate, payRef, paidUsdc: 0,
    _gpu: gpu, _gpuSpec: { cardId: gpu.cardId, cardUuid: gpuCards[gpu.cardId]?.uuid || null, vramCapGb: gpu.vramGb, computeShare: gpu.computeShare },
    _port: 0, _sshPort: 0, _sshKeySource: keySource, _authorizedKey: authorizedKey, _payTimer: null,
  };
  deployments.set(id, rec);
  payRefIndex.set(payRef.toLowerCase(), id);
  armPayTimer(rec);            // release the reservation if unpaid by payDeadline
  saveStateSoon();

  // AUTO_PROVISION: boot now without an on-chain payment (manual billing / pilot).
  if (AUTO_PROVISION) {
    if (!(await forceProvision(rec)))
      return fail(res, 502, "provision_failed", rec.error || "provisioning failed");
    console.log(`[auto-provision] ${id} booted without payment; `
              + `remaining=${rec.remainingMs != null ? Math.round(rec.remainingMs / 1000) + "s" : "unlimited"}`);
  }

  const out = view(rec);                                  // includes payment instructions + ssh
  if (oneTimePrivateKey) out.ssh.privateKey = oneTimePrivateKey; // shown once; never persisted
  res.status(201).json(out);
});

// Spawn the tenant's MPS-capped worker process (called once, on first payment).
async function provisionTenant(rec) {
  try {
    const sp = await spawnContainer({ deploymentId: rec.id, gpu: rec._gpuSpec,
      image: rec.image, share: rec.resources.share, appPort: rec.network.port, ports: rec.firewall });
    rec._port = sp.internalPort; rec._sshPort = sp.sshPort;
    if (sp.vmId) { rec._vmId = sp.vmId; rec._vmHostPort = sp.hostPort; }
    if (sp.portMap) rec.portMap = sp.portMap;   // logical -> actual (public: clients see their mapping)
    if (!rec.startedAt) rec.startedAt = Date.now();
    rec.status = "running"; rec.paused = false; rec.pauseReason = null; rec._lastTickAt = Date.now();
    return true;
  } catch (e) {
    rec.status = "failed"; rec.error = e.message;
    if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
    console.error(`[provision] ${rec.id} failed: ${e.message}`);
    return false;
  }
}

// Provision a deployment WITHOUT a payment (auto-provision / admin). Clears the
// unpaid-reservation timer and sets the optional safety expiry.
async function forceProvision(rec) {
  if (rec._payTimer) { clearTimeout(rec._payTimer); rec._payTimer = null; }
  const ok = await provisionTenant(rec);
  if (ok) rec.remainingMs = AUTO_PROVISION_HOURS > 0 ? AUTO_PROVISION_HOURS * 3600 * 1000 : null; // null = unlimited
  saveStateSoon();
  return ok;
}

// A Paid event landed: provision on first payment, extend expiry on top-ups.
async function onPaid(payRefHex, payer, amountRaw) {
  const id = payRefIndex.get(String(payRefHex).toLowerCase());
  if (!id) { console.warn(`[pay] payment for unknown ref ${payRefHex} (${amountRaw})`); return; }
  const rec = deployments.get(id); if (!rec) return;
  const seconds = usdcToSeconds(amountRaw, rec.rate);
  rec.paidUsdc = (rec.paidUsdc || 0) + Number(amountRaw);
  if (rec.status === "awaiting_payment") {
    if (rec._payTimer) { clearTimeout(rec._payTimer); rec._payTimer = null; }
    if (!(await provisionTenant(rec))) { saveStateSoon(); return; }  // failed provisioning surfaces in the record
    rec.remainingMs = seconds * 1000;
    console.log(`[pay] ${id} funded ${(Number(amountRaw)/1e6).toFixed(2)} USDC -> +${Math.round(seconds)}s, provisioned`);
  } else if (rec.status === "running") {
    // top-up adds to the balance; a grace overrun (negative balance) is forgiven.
    // remainingMs === null (unlimited pilot) stays unlimited.
    if (rec.remainingMs != null) rec.remainingMs = Math.max(0, rec.remainingMs) + seconds * 1000;
    console.log(`[pay] ${id} top-up ${(Number(amountRaw)/1e6).toFixed(2)} USDC -> +${Math.round(seconds)}s (${timeRemainingSec(rec) ?? "unlimited"}s left)`);
  } else {
    console.warn(`[pay] ${id} payment ${(Number(amountRaw)/1e6).toFixed(2)} USDC but status=${rec.status}; ignored (no refunds in pay-per-deploy)`);
  }
  saveStateSoon();
}

// Watch the forwarder for Paid events (poll getLogs; robust on public RPC).
// Robust log watch on a public RPC. Two failure modes the naive "scan to tip,
// advance past it" loop hits, and how we kill both:
//  (1) MISSED LOGS -> lost payments. mainnet.base.org is load-balanced; getBlockNumber()
//      and getLogs() can hit different nodes at different heights, so a log in the newest
//      blocks can be absent from the response — and advancing past it drops the payment
//      forever. Fix: only finalize up to tip - PAY_CONFIRMATIONS, and re-scan a trailing
//      overlap every poll so a momentarily-behind node gets a second look.
//  (2) DOUBLE-CREDIT. Re-scanning would re-run onPaid (a top-up) for the same event.
//      Fix: dedup on txHash:logIndex — each payment log is handled exactly once, which
//      also makes a mid-poll RPC failure safe to retry.
const PAY_CONFIRMATIONS = parseInt(process.env.PAY_CONFIRMATIONS || "3", 10);    // blocks of lag before trusting a log
const PAY_RESCAN_BLOCKS = parseInt(process.env.PAY_RESCAN_BLOCKS || "20", 10);   // trailing overlap re-scanned each poll (~40s on Base)
const PAY_CHUNK_BLOCKS  = BigInt(process.env.PAY_CHUNK_BLOCKS || "4000");        // max getLogs range per call (public-RPC safe)
const PAY_MAX_CATCHUP   = BigInt(process.env.PAY_MAX_CATCHUP_BLOCKS || "200000"); // ~4.6 days of Base blocks
const _seenLogs = new Map();   // "txHash:logIndex" -> blockNumber (pruned as the window advances)
let _payFromBlock = null;      // persisted: after downtime the watcher resumes HERE, not at the tip
let _lastPollOkAt = 0;         // freshness signal: billing + reservation expiry freeze while the watcher is blind
let _polling = false;
async function pollPayments() {
  if (!FORWARDER_ADDRESS || _polling) return;   // no overlap: catch-up after downtime can outlast one poll interval
  _polling = true;
  try {
    // instructions need the token's EIP-712 domain; retry here until the first read lands
    if (!_usdcDomain) refreshUsdcDomain().catch(() => {});
    // retry ETH payments that missed an oracle read (never lose a payment)
    if (_pendingEth.length) {
      const q = _pendingEth.splice(0);
      for (const p of q) await onPaidEth(p.payRefHex, p.payer, p.wei);
    }
    const tip = await chainClient.getBlockNumber();
    const safe = tip - BigInt(PAY_CONFIRMATIONS);                    // don't finalize logs newer than this
    if (safe < 0n) return;
    if (_payFromBlock == null) _payFromBlock = safe + 1n;            // first EVER run: start at the (confirmed) tip
    if (safe < _payFromBlock) { _lastPollOkAt = Date.now(); return; } // no new confirmed blocks yet
    if (safe - _payFromBlock > PAY_MAX_CATCHUP) {                    // bound a very long outage
      console.warn(`[pay] catch-up clamped: ${safe - _payFromBlock} blocks behind, scanning last ${PAY_MAX_CATCHUP}`);
      _payFromBlock = safe - PAY_MAX_CATCHUP;
    }
    // Walk the window in chunks: after an outage it can span hours of blocks, and
    // public RPCs reject or silently truncate huge ranges. _lastPollOkAt stays
    // stale until fully caught up, so clocks stay frozen while payments made
    // during the outage are still being credited.
    while (_payFromBlock <= safe) {
      const from = _payFromBlock > BigInt(PAY_RESCAN_BLOCKS) ? _payFromBlock - BigInt(PAY_RESCAN_BLOCKS) : 0n;
      const to = (safe - from) > PAY_CHUNK_BLOCKS ? from + PAY_CHUNK_BLOCKS : safe;
      for (const [k, b] of _seenLogs) if (b < from) _seenLogs.delete(k);   // prune dedup set below the window
      for (const [evt, isEth] of [[PAY_EVENT, false], [PAY_ETH_EVENT, true]]) {
        const logs = await chainClient.getLogs({ address: getAddress(FORWARDER_ADDRESS),
          event: evt, fromBlock: from, toBlock: to });
        for (const lg of logs) {
          const key = `${lg.transactionHash}:${lg.logIndex}`;
          if (_seenLogs.has(key)) continue;                          // exactly-once, even across re-scans / partial failures
          _seenLogs.set(key, lg.blockNumber);
          const a = lg.args || {};
          if (isEth) await onPaidEth(a.deploymentId, a.payer, a.amountWei);
          else       await onPaid(a.deploymentId, a.payer, a.amount);
        }
      }
      _payFromBlock = to + 1n;
      saveStateSoon();                                               // persist the cursor: a restart resumes, not skips
    }
    _lastPollOkAt = Date.now();
  } catch (e) { console.warn(`[pay] poll error: ${e.shortMessage || e.message}`); }
  finally { _polling = false; }
}
function startPaymentWatcher() {
  if (!FORWARDER_ADDRESS) { console.warn("[pay] FORWARDER_ADDRESS unset - payments disabled (deployments will sit awaiting_payment)"); return; }
  console.log(`[pay] watching ${FORWARDER_ADDRESS} for Paid + PaidEth events every ${PAY_POLL_SEC}s (ETH priced via ${ETH_USD_FEED})`);
  const t = setInterval(pollPayments, PAY_POLL_SEC * 1000); if (t.unref) t.unref();
  pollPayments();
  // prime the USDC EIP-712 domain so payment instructions can carry it (retries in pollPayments)
  refreshUsdcDomain().catch((e) => console.warn(`[pay] USDC domain read: ${e.shortMessage || e.message}`));
  // prime + keep the ETH/USD cache warm so payment instructions can quote ethUsd
  ethUsdPrice8().catch((e) => console.warn(`[pay] ETH/USD feed: ${e.shortMessage || e.message}`));
  const p = setInterval(() => ethUsdPrice8().catch(() => {}), 300_000); if (p.unref) p.unref();
}

// ---- fair-billing ticker (replaces the wall-clock reaper) -------------------
// Drains each running deployment's remainingMs by REAL elapsed time, but ONLY
// while the platform is serving: backend manager healthy, payment watcher fresh,
// app instance actually alive. Otherwise the deployment is marked paused and its
// clock FREEZES; it resumes on the first healthy tick. Supervisor downtime
// freezes too: ticks simply don't happen while we're down, and loadState()
// resets _lastTickAt on boot so the gap is never charged.
const isMock = () => /^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "");
async function backendHealthy() {
  if (isMock()) return true;
  try { PROVISION_BACKEND === "vm" ? await vmHealth() : await mgrHealth(); return true; }
  catch { return false; }
}
// vm backend: is this deployment's app instance still alive in the manager?
// (the wasm-manager runs apps as subprocesses; a manager restart loses them)
async function instanceAlive(rec) {
  if (isMock() || PROVISION_BACKEND !== "vm") return true;  // worker backend: manager health is the best signal we have
  if (!rec._vmId) return false;
  const r = await vmReq("GET", `/vms/${encodeURIComponent(rec._vmId)}`, null, 5000).catch(() => null);
  if (!r) return false;                        // manager unreachable mid-tick: freeze, don't respawn
  return r.status === 200;
}
function pauseRec(rec, reason) {
  if (!rec.paused || rec.pauseReason !== reason) {
    console.warn(`[bill] ${rec.id} clock FROZEN (${reason})`);
    rec.paused = true; rec.pauseReason = reason; saveStateSoon();
  }
  rec._lastTickAt = Date.now();                // downtime is never charged
}
function resumeRec(rec) {
  if (rec.paused) {
    console.log(`[bill] ${rec.id} clock resumed (${timeRemainingSec(rec) ?? "unlimited"}s left)`);
    rec.paused = false; rec.pauseReason = null;
    rec._lastTickAt = Date.now();              // bill from the moment of resume, not the frozen span
    saveStateSoon();
  }
}
// Respawn a still-funded app whose instance vanished (e.g. the manager container
// restarted). Never marks the record failed - the user keeps their frozen
// balance; retries back off so a broken image can't hammer the manager.
async function respawnTenant(rec) {
  if (rec._respawning || Date.now() < (rec._respawnAt || 0)) return false;
  rec._respawning = true;
  try {
    const sp = await spawnContainer({ deploymentId: rec.id, gpu: rec._gpuSpec,
      image: rec.image, share: rec.resources.share, appPort: rec.network.port, ports: rec.firewall });
    rec._port = sp.internalPort; rec._sshPort = sp.sshPort;
    if (sp.vmId) { rec._vmId = sp.vmId; rec._vmHostPort = sp.hostPort; }
    if (sp.portMap) rec.portMap = sp.portMap;
    rec._respawnAt = 0; rec._respawnBackoffMs = 0;
    console.log(`[bill] ${rec.id} instance respawned after outage`);
    return true;
  } catch (e) {
    rec._respawnBackoffMs = Math.min((rec._respawnBackoffMs || 15000) * 2, 300000);
    rec._respawnAt = Date.now() + rec._respawnBackoffMs;
    console.warn(`[bill] ${rec.id} respawn failed (${e.message}); retry in ${Math.round(rec._respawnBackoffMs / 1000)}s`);
    return false;
  } finally { rec._respawning = false; }
}
function startBillingTicker() {
  const t = setInterval(async () => {
    const now = Date.now();
    const healthy = await backendHealthy();
    const watcherOk = !FORWARDER_ADDRESS || (now - _lastPollOkAt) < WATCHER_STALE_SEC * 1000;
    for (const rec of deployments.values()) {
      if (rec.status !== "running") continue;
      if (!healthy)   { pauseRec(rec, "backend_down");  continue; }
      if (!watcherOk) { pauseRec(rec, "watcher_stale"); continue; }
      if (!(await instanceAlive(rec))) {
        pauseRec(rec, "instance_missing");
        if (await respawnTenant(rec)) resumeRec(rec);
        continue;
      }
      resumeRec(rec);
      if (rec.remainingMs == null) { rec._lastTickAt = now; continue; }   // unlimited (pilot)
      // clamp so an event-loop stall or clock jump can't overcharge one tick
      const elapsed = Math.min(Math.max(0, now - (rec._lastTickAt || now)), 2 * BILL_TICK_SEC * 1000);
      rec._lastTickAt = now;
      rec.remainingMs -= elapsed;
      rec.consumedMs = (rec.consumedMs || 0) + elapsed;
      saveStateSoon();
      if (rec.remainingMs < -GRACE_SEC * 1000) {
        console.log(`[reaper] ${rec.id} out of funded time -> teardown`);
        try { await stopContainer(rec); } catch {}
        if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
        rec.status = "expired";
      }
    }
    if (_stateDirty) saveStateNow();
  }, BILL_TICK_SEC * 1000);
  if (t.unref) t.unref();
}

app.get("/v1/deployments", authed, (req, res) =>
  res.json({ data: [...deployments.values()].filter(d => d.owner === req.address).map(view), cursor: null }));

app.get("/v1/deployments/:id", authed, (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  res.json(view(rec));
});

// Operator-only: provision an awaiting_payment deployment WITHOUT a payment.
// Gated by ADMIN_TOKEN (x-admin-token header); returns 404 if the token is unset
// or wrong, so the endpoint is invisible without it. Use for manually-billed deploys.
app.post("/v1/admin/deployments/:id/provision", async (req, res) => {
  if (!ADMIN_TOKEN || req.headers["x-admin-token"] !== ADMIN_TOKEN)
    return fail(res, 404, "not_found", "Not found.");
  const rec = deployments.get(req.params.id);
  if (!rec) return fail(res, 404, "not_found", "No such deployment.");
  if (rec.status !== "awaiting_payment")
    return fail(res, 409, "not_provisionable", `Deployment is ${rec.status}.`);
  if (!(await forceProvision(rec)))
    return fail(res, 502, "provision_failed", rec.error || "provisioning failed");
  console.log(`[admin] ${rec.id} provisioned by operator (no payment)`);
  res.json(view(rec));
});

app.delete("/v1/deployments/:id", authed, async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  if (rec._payTimer) { clearTimeout(rec._payTimer); rec._payTimer = null; }
  try { await stopContainer(rec); } catch {}
  if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; }
  rec.status = "stopping";
  saveStateSoon();
  res.json({ id: rec.id, status: "stopping",
             paidUsdc: ((rec.paidUsdc || 0) / 1e6).toFixed(2),
             ranSeconds: Math.round((rec.consumedMs || 0) / 1000),
             note: "Pay-per-deploy: no balance is held, so unused funded time is forfeit on early stop." });
});

// Top-up instructions: just call NanPay.pay(deploymentRef, amount) again to extend.
app.post("/v1/deployments/:id/topup", authed, (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  if (!["running", "awaiting_payment"].includes(rec.status))
    return fail(res, 409, "not_toppable", `Deployment is ${rec.status}.`);
  res.json({ id: rec.id, status: rec.status, timeRemainingSec: timeRemainingSec(rec), payment: paymentInstructions(rec) });
});

// Optional ?nonce=<64 hex chars>: freshness challenge folded into the GPU report
// (the TDX quote needs none - it binds the long-lived TLS key, and freshness
// comes from fetching the RAD over your own connection).
function attestNonce(req, res) {
  const n = req.query.nonce;
  if (n == null) return randomBytes(32).toString("hex");
  if (!/^[0-9a-fA-F]{64}$/.test(n)) { fail(res, 422, "bad_nonce", "nonce must be 32 bytes of hex."); return null; }
  return n.toLowerCase();
}
app.get("/v1/deployments/:id/attestation", authed, async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  const nonce = attestNonce(req, res); if (nonce == null) return;
  try { res.json({ deploymentId: rec.id, generatedAt: new Date().toISOString(),
                   ...(await getMeasurements(rec, { origin: originOf(req), nonce })),
                   guideUrl: "https://nan.host/#attest" }); }
  catch (e) { fail(res, 502, "attestation_error", e.message); }
});

// Enclave-level attestation, PUBLIC: verify the enclave before logging in or
// sending a byte. GPU evidence is included from a short cache (refreshed with a
// self-chosen nonce) so an unauthenticated caller can't spam NVML report
// generation; pass a nonce on the per-deployment endpoint for a fresh challenge.
let _gpuEvCache = null;                       // { ev?, err?, at } - failures cached briefly too
app.get("/v1/attestation", async (req, res) => {
  const out = await getMeasurements(null, { origin: originOf(req) });
  try {
    const ttl = _gpuEvCache?.err ? 10_000 : 60_000;
    if (!_gpuEvCache || Date.now() - _gpuEvCache.at > ttl) {
      try { _gpuEvCache = { ev: await fetchGpuEvidence(randomBytes(32).toString("hex"), 15000), at: Date.now() }; }
      catch (e) { _gpuEvCache = { err: e, at: Date.now() }; }
    }
    if (_gpuEvCache.err) throw _gpuEvCache.err;
    const ev = _gpuEvCache.ev;
    out.gpu = { technology: "nvidia-cc", ccMode: ev.ccMode ?? null, nonce: ev.nonce,
                driverVersion: ev.driverVersion ?? null, generatedAt: new Date(_gpuEvCache.at).toISOString(),
                report: ev.gpus?.[0]?.attestationReport_b64 ?? null,
                certChain: ev.gpus?.[0]?.attestationCertChain_b64 ?? null, gpus: ev.gpus || [] };
  } catch (e) { out.gpu = { technology: "nvidia-cc", available: false, error: e.message }; }
  res.json({ generatedAt: new Date().toISOString(), ...out, guideUrl: "https://nan.host/#attest" });
});

// Tail the worker's stdout/stderr (owner only). ?tail=N (default 200, max 2000).
app.get("/v1/deployments/:id/logs", authed, async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  if (/^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) return res.type("text/plain").send("[mock] no real worker; logs unavailable\n");
  const tail = String(Math.min(2000, Math.max(1, parseInt(req.query.tail, 10) || 200)));
  try {
    const r = await dockerReq("GET", `/containers/${containerName(rec.id)}/logs?stdout=1&stderr=1&tail=${tail}`, null, 15000);
    if (r.status >= 400) return fail(res, 502, "logs_error", r.buf.toString().slice(0, 200));
    res.type("text/plain").send(demuxLogs(r.buf));
  } catch (e) { fail(res, 502, "logs_error", (e.message || "").toString().slice(0, 300)); }
});

app.use((_req, res) => fail(res, 404, "not_found", "No such route."));
await initGpu();
await initMps();
await initSshHostKey();
// restore persisted deployments/payment cursor BEFORE serving traffic or polling:
// the downtime gap is frozen (never charged) and reservations shift by the gap.
initStatePersistence();
loadState();

// ---------------------------------------------------------------------------
// SSH TUNNEL - ssh rides the one attested origin as a WebSocket at /x/:id/ssh.
// `websocat -b` carries the raw SSH byte stream; we bridge it to the per-
// deployment sshd the SUPERVISOR hosts (measured host key from initSshHostKey;
// the sandbox image needs no sshd). Same gate as the data path: session JWT
// (Authorization header or ?token= for browsers/websocat) + ownership. No second
// external port is opened.
// ---------------------------------------------------------------------------
const server = http.createServer(app);
const wss = new WebSocketServer({ noServer: true });

async function authUpgrade(req) {
  let token = null;
  const h = (req.headers.authorization || "").match(/^Bearer\s+(.+)$/i);
  if (h) token = h[1];
  else { try { token = new URL(req.url, "http://x").searchParams.get("token"); } catch {} }
  if (!token) return null;
  try { const { payload } = await jwtVerify(token, SECRET); return getAddress(payload.sub); } catch { return null; }
}

// bridge a WebSocket to a local TCP port, binary frames both ways
function wsTcpBridge(req, socket, head, port) {
  wss.handleUpgrade(req, socket, head, (ws) => {
    const tcp = net.connect(port, "127.0.0.1");
    const close = () => { try { ws.close(); } catch {} try { tcp.destroy(); } catch {} };
    tcp.on("connect", () => {
      ws.on("message", (d) => tcp.write(d));
      tcp.on("data", (d) => { if (ws.readyState === ws.OPEN) ws.send(d); });
    });
    ws.on("close", close); ws.on("error", close);
    tcp.on("close", close); tcp.on("error", close);
  });
}

server.on("upgrade", async (req, socket, head) => {
  const deny = (line) => { socket.write(`HTTP/1.1 ${line}\r\n\r\n`); socket.destroy(); };

  // ---- app TCP ports: /x/:id/tcp/:port — the declared-firewall data path ----
  // Auth follows the deployment's `public` flag (like the HTTP path). Two gates
  // before bridging: the port must be DECLARED in the firewall config, and the
  // manager must confirm THIS app actually bound it (boundPorts) — otherwise a
  // tenant could declare-but-not-bind a port and bridge into a sibling's socket.
  const t = (req.url || "").match(/^\/x\/([^/?]+)\/tcp\/(\d{1,5})(?:\?|$)/);
  if (t) {
    const rec = deployments.get(t[1]), port = +t[2];        // `port` is the LOGICAL port (the app's advertised one)
    if (!rec)                                 return deny("404 Not Found");
    if (!fwTcpPorts(rec).includes(port))      return deny("404 Not Found");
    if (!rec.public) {
      const addr = await authUpgrade(req);
      if (!addr)              return deny("401 Unauthorized");
      if (rec.owner !== addr) return deny("403 Forbidden");
    }
    if (rec.status !== "running" || !rec._vmId) return deny("409 Conflict");
    // resolve logical -> actual bind for THIS deployment (two tenants can both be
    // "the 5432 app"; each has its own actual port), then confirm the app bound it.
    const actual = (rec.portMap && rec.portMap["tcp:" + port]) || port;
    const vr = await vmReq("GET", `/vms/${encodeURIComponent(rec._vmId)}`).catch(() => null);
    const bound = (vr && vr.body && vr.body.boundPorts) || [];
    if (!bound.includes(actual)) return deny("409 Conflict");   // app hasn't bound it (yet)
    return wsTcpBridge(req, socket, head, actual);
  }

  // ---- SSH: always owner-only, regardless of `public` ----
  const m = (req.url || "").match(/^\/x\/([^/?]+)\/ssh(?:\?|$)/);
  if (!m) { socket.destroy(); return; }
  const rec  = deployments.get(m[1]);
  const addr = await authUpgrade(req);
  if (!rec || !rec._sshPort)     return deny("404 Not Found");
  if (!addr)                     return deny("401 Unauthorized");
  if (rec.owner !== addr)        return deny("403 Forbidden");
  if (rec.status !== "running")  return deny("409 Conflict");
  wsTcpBridge(req, socket, head, rec._sshPort);
});

server.listen(PORT, () => console.log(`nan supervisor on :${PORT} · ${GPU_COUNT}×GPU @ ${CARD_VRAM_GB}GB (arbitrary split) · ssh host key ${SSH_HOST_KEY_FP}`));

// advertise this enclave on-chain (opt-in, non-blocking, never fatal)
registerOnChain();

// pay-per-deploy: watch the forwarder for payments + fair-billing ticker (drains
// funded time only while healthy; freezes through outages; reaps at -grace)
startPaymentWatcher();
startBillingTicker();
