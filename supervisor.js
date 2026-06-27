// NAN supervisor — the WHOLE service, running INSIDE the Tinfoil enclave behind
// the shim (the single ingress). It is the measured/attested image: the same
// published code that checks a user's signature, gates on escrow, mints the
// session token, launches the per-use container, and proxies the data path.
//
// There is no external tier. Browser -> shim -> here, for BOTH control and data:
//   control:  /v1/*        (SIWE login, deployments, account, attestation)
//   data:     /x/:id/*     (verify session token + ownership, proxy to the
//                           spawned container; fly.io used to do nothing here —
//                           now nothing external touches a prompt at all)
//
// One signing SECRET (an enclave secret). One token type: the session JWT the
// browser gets at login is reused as the capability on the data path.
//
// >>> The ONLY thing left to implement for your CVM is spawn/stop/measure below.

import express from "express";
import cors from "cors";
import http from "node:http";
import net from "node:net";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { WebSocketServer } from "ws";
import { verifyMessage, createPublicClient, http as viemHttp, getAddress } from "viem";
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
const ESCROW_ENABLED = /^(1|true|on)$/i.test(process.env.ESCROW_ENABLED || "");
const ESCROW_ADDRESS = process.env.ESCROW_ADDRESS || "";
const BASE_RPC       = process.env.BASE_RPC || "https://mainnet.base.org";
const SESSION_TTL    = parseInt(process.env.SESSION_TTL || "43200", 10); // 12h: long enough to cover a deployment's data-path use
const SSH_USER       = process.env.SSH_USER || "instance"; // login user the supervisor's sshd drops into
const DEFAULT_IMAGE  = process.env.DEFAULT_IMAGE || "debian:bookworm-slim"; // any stock image; sshd is hosted by the supervisor, not the image
// The sandbox sshd host key is GENERATED ONCE AT BOOT inside the enclave and
// measured into a TDX RTMR (see initSshHostKey) — so its fingerprint is
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
const MIN_COMPUTE_SHARE = parseFloat(process.env.MIN_COMPUTE_SHARE || "0.1428571"); // default/floor compute share (1/7)
const GRANULARITY_GB  = parseFloat(process.env.VRAM_GRANULARITY_GB || "1"); // request rounding; 1 GB ≈ arbitrary

const round1 = (x) => Math.round(x * 10) / 10;
const round3 = (x) => Math.round(x * 1000) / 1000;
const memShareOf = (vramGb) => vramGb / CARD_VRAM_GB;

// per-card free pools (vram + compute). With CC on there is exactly one whole
// device per card — no MIG instances to enumerate.
const gpuCards = Array.from({ length: GPU_COUNT }, (_, i) => ({ id: i, uuid: null, vramFree: CARD_VRAM_GB, computeFree: 1 }));

// price = whole-card rate × the LARGER of memory share or compute share.
function rateFor(vramGb, computeShare) {
  if (!(vramGb > 0)) return CPU_RATE;
  return FULL_RATE * Math.max(memShareOf(vramGb), computeShare);
}
// normalize a request: round VRAM up to granularity; default/clamp compute share.
function normalizeReq(vramGb, computeShare) {
  const v = Math.ceil(vramGb / GRANULARITY_GB) * GRANULARITY_GB;
  let c = (computeShare == null) ? Math.max(MIN_COMPUTE_SHARE, memShareOf(v)) : computeShare;
  c = Math.min(1, Math.max(0, c));
  return { vramGb: v, computeShare: c };
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

async function initGpu() {
  // TODO: discover card UUIDs (`nvidia-smi -L`) → gpuCards[i].uuid, so each worker
  // can be pinned with CUDA_VISIBLE_DEVICES=<uuid>. Optionally read true usable
  // VRAM (`nvidia-smi --query-gpu=memory.total`) to set CARD_VRAM_GB. With CC on,
  // each card is one whole device — there are no MIG instances to list.
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
    // TODO: extend SSH_HOST_KEY_FP (or the raw pubkey) into a TDX RTMR here so
    // getMeasurements() reports a measured, not just asserted, host key.
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
// >>> IMPLEMENT THESE for your CVM launch mechanism (docker socket / nested
//     microVM / namespaces). Contract: one ingress port, no sibling reach,
//     and BEFORE launch extend a TDX RTMR with image.digest so getMeasurements()
//     is honest. If the CVM can't extend an RTMR from the guest, attestation
//     covers this supervisor only — say so in /attestation rather than implying
//     the user image is measured.
//     SSH: the sandbox runs ANY stock image and needs NO sshd of its own. The
//     supervisor hosts sshd (measured host key from initSshHostKey); spawn starts
//     a loopback sshd for this deployment using SSH_HOST_KEY_PATH, installs
//     `authorizedKey`, and sets a ForceCommand that exec's into THIS sandbox's
//     namespace. Return its loopback port as sshPort.
// ============================================================================
async function spawnContainer({ deploymentId, owner, image, command, env, port, gpu, budget, authorizedKey }) {
  // MOCK mode: no real CVM launch yet — return fake ports so the WHOLE control
  // plane (auth, pricing, availability, deployment CRUD, billing) works end-to-end
  // while spawn/stop/measure are still stubs. Set MOCK_SPAWN=1 to enable.
  if (/^(1|true|on)$/i.test(process.env.MOCK_SPAWN || "")) {
    console.log(`[mock] spawn ${deploymentId} image=${image?.reference} gpu=${gpu ? gpu.vramCapGb + "GB@" + gpu.computeShare : "cpu"}`);
    return { internalPort: port || 8080, sshPort: 0 };
  }
  // TODO: launch image.reference (pinned by image.digest) as this tenant's OWN
  // worker PROCESS — never shared with another tenant. If gpu != null:
  //   • pin the card:        CUDA_VISIBLE_DEVICES = gpu.cardUuid
  //   • cap VRAM:            enforce gpu.vramCapGb in the worker's allocator
  //                          (reject allocs past the cap; the supervisor already
  //                          reserved gpu.vramCapGb + overhead against the card)
  //   • cap compute:         run under MPS with
  //                          CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = round(gpu.computeShare*100)
  // Isolation is the process boundary; rotating a tenant = killing this process
  // (the driver scrubs its VRAM on exit). Then start the loopback sshd:
  //   -h ${SSH_HOST_KEY_PATH}            (the measured, boot-generated host key)
  //   authorized_keys = authorizedKey
  //   ForceCommand = exec into this worker's namespace as ${SSH_USER}
  // return { internalPort, sshPort };
  throw new Error("spawnContainer() not implemented — wire to your CVM runtime");
}
async function stopContainer(rec)  { /* TODO: SIGKILL the worker; its VRAM is scrubbed on exit and the slice is freed in the route */ }
async function getMeasurements(rec) {
  // TODO: return the live TDX quote (+ whole-card NVIDIA CC report) folding in image.digest.
  return {
    tlsKeyFingerprint: "sha256:<enclave-tls-pubkey-hash>",
    sshHostKeyFingerprint: SSH_HOST_KEY_FP, // boot-generated; measured into an RTMR (see initSshHostKey TODO)
    vm:  { technology: "intel-tdx", quote: "<base64 tdx quote>", measurements: { rtmr3: rec.digest }, verified: true },
    gpu: rec._gpu ? { technology: "nvidia-cc", ccMode: "on", vramCapGb: rec.resources.vramGb,
                      computeShare: rec.resources.computeShare, report: "<base64 nvidia report>", verified: true } : null,
  };
}
function capacity() {
  return {
    cpu: 64,
    gpu: gpuCards.map(c => ({ id: c.id, vramFreeGb: round1(c.vramFree), computeFree: round3(c.computeFree) })),
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
// DATA PATH — registered BEFORE express.json() so the body streams untouched.
// Same token, same origin as control; supervisor checks ownership, then proxies.
// ---------------------------------------------------------------------------
app.use("/x/:id", async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || !rec._port) return fail(res, 404, "not_found", "Unknown deployment.");
  const addr = await addrFromAuth(req);
  if (!addr) return fail(res, 401, "unauthorized", "Missing or invalid token.");
  if (rec.owner !== addr) return fail(res, 403, "forbidden", "Not your deployment.");
  if (rec.status !== "running") return fail(res, 409, "not_running", `Deployment is ${rec.status}.`);

  const headers = { ...req.headers, host: `127.0.0.1:${rec._port}` };
  delete headers.authorization; // the NAN token stays at the supervisor; container never sees it
  const up = http.request({ host: "127.0.0.1", port: rec._port, method: req.method, path: req.url, headers }, (upRes) => {
    res.writeHead(upRes.statusCode || 502, upRes.headers); upRes.pipe(res);
  });
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
app.get("/v1/health", (_req, res) => res.json({ status: "ok", deployments: deployments.size }));
app.get("/v1/version", (_req, res) => res.json({ service: "nan-supervisor/0.1.0", contract: "nan-openapi/1.0.0", chainId: CHAIN_ID }));

app.get("/v1/pricing", (_req, res) => res.json({
  assets: ["ETH","USDC"],
  model: "Request any VRAM slice (GB) and an optional compute share (0–1) of a card. Both are software-enforced caps — CC disables MIG, so splits are arbitrary, not fixed profiles. Billed per second by the LARGER of memory share (vramGb / cardVramGb) or compute share, times the whole-card rate.",
  card: { vramGb: CARD_VRAM_GB, count: GPU_COUNT,
          wholeCardPerSecondUsdc: FULL_RATE.toFixed(7), wholeCardPerHourUsdc: (FULL_RATE * 3600).toFixed(2) },
  cpu: { ratePerSecondUsdc: CPU_RATE.toFixed(7), ratePerHourUsdc: (CPU_RATE * 3600).toFixed(2) },
  minComputeShare: round3(MIN_COMPUTE_SHARE),
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

app.get("/availability", (_req, res) => {
  const c = capacity();
  res.json({
    cpu: { available: c.cpu, status: "available" },
    gpu: c.gpu,
    vramFreeGb: c.vramFreeGb,
    maxVramGb: c.maxVramGb,
    maxComputeShare: c.maxComputeShare,
    cardVramGb: CARD_VRAM_GB, cards: GPU_COUNT,
    updatedAt: new Date().toISOString(),
  });
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
// account / escrow (outbound Base RPC — confirm CVM egress allows BASE_RPC)
// ============================================================================
const ESCROW_ABI = [{ type: "function", name: "available", stateMutability: "view",
  inputs: [{ name: "account", type: "address" }, { name: "asset", type: "uint8" }],
  outputs: [{ name: "", type: "uint256" }] }]; // match to SealCoordinator once finalized

async function readEscrow(address) {
  if (!ESCROW_ENABLED || !ESCROW_ADDRESS)
    return [{ asset: "USDC", deposited: "0", reserved: "0", available: "999999.00" }]; // dev: treat as funded
  const usdc = await chainClient.readContract({ address: getAddress(ESCROW_ADDRESS), abi: ESCROW_ABI,
    functionName: "available", args: [getAddress(address), 0] });
  const avail = (Number(usdc) / 1e6).toFixed(2);
  return [{ asset: "USDC", deposited: avail, reserved: "0", available: avail }];
}

app.get("/v1/account", authed, async (req, res) => {
  try {
    const balances = await readEscrow(req.address);
    const mine = [...deployments.values()].filter(d => d.owner === req.address);
    res.json({ address: req.address, escrow: { contract: ESCROW_ADDRESS || null, chainId: CHAIN_ID }, balances,
               deployments: { active: mine.filter(d => d.status === "running").length, total: mine.length } });
  } catch (e) { fail(res, 502, "escrow_error", e.message); }
});

// ============================================================================
// deployments
// ============================================================================
const spentOf = (rec) => {
  if (!rec.startedAt) return "0.00";
  const cap = parseFloat(rec.budget.limit);
  const raw = ((Date.now() - rec.startedAt) / 1000) * (rec.rate || 0);
  return Math.min(raw, cap).toFixed(2);
};
const view = (rec) => { const o = { ...rec }; delete o._port; delete o._gpu; delete o.rate;
                        delete o._sshPort; delete o._sshKeySource; delete o._authorizedKey;
                        o.ssh = sshAccessOf(rec);
                        o.budget = { ...rec.budget, spent: spentOf(rec) }; return o; };

app.post("/v1/deployments", authed, async (req, res) => {
  const b = req.body || {};
  if (!b.budget || !b.budget.asset || !b.budget.limit) return fail(res, 422, "invalid_spec", "budget {asset, limit} is required.");
  const image = (b.image && b.image.reference) ? b.image : { reference: DEFAULT_IMAGE };
  const appPort = Number(b.port) || 8080;
  if (b.sshPublicKey != null && !/^(ssh-ed25519|ssh-rsa|ecdsa-sha2-|sk-ssh-|sk-ecdsa-)/.test(String(b.sshPublicKey).trim()))
    return fail(res, 422, "invalid_spec", "sshPublicKey must be an OpenSSH public key (ssh-ed25519 / ssh-rsa / ecdsa / sk-*).");
  const vramGb0 = Number((b.resources && b.resources.vramGb) || 0);
  if (!(vramGb0 >= 0) || vramGb0 > 100000) return fail(res, 422, "invalid_spec", "resources.vramGb out of range.");
  let computeShare0 = (b.resources && b.resources.computeShare != null) ? Number(b.resources.computeShare) : null;
  if (computeShare0 != null && !(computeShare0 > 0 && computeShare0 <= 1))
    return fail(res, 422, "invalid_spec", "resources.computeShare must be in (0, 1].");

  // allocate an ARBITRARY GPU slice (vramGb + compute share); 0 VRAM => CPU-only
  let gpu = null, rate = CPU_RATE, slice = null;
  if (vramGb0 > 0) {
    slice = normalizeReq(vramGb0, computeShare0);
    if (slice.vramGb > maxFreeVram() + 1e-9)
      return fail(res, 422, "invalid_spec", `requested ${slice.vramGb}GB VRAM exceeds the largest free slice (${round1(maxFreeVram())}GB on a ${CARD_VRAM_GB}GB card).`);
    gpu = allocGpu(slice.vramGb, slice.computeShare);
    if (!gpu) return fail(res, 409, "no_capacity",
      `No single card has ${slice.vramGb}GB VRAM and ${round3(slice.computeShare)} compute share free together (max free: ${round1(maxFreeVram())}GB, ${round3(maxFreeCompute())} share).`);
    rate = rateFor(slice.vramGb, slice.computeShare);
  }
  const release = () => { if (gpu) { releaseGpu(gpu); gpu = null; } };

  try {
    const bal = await readEscrow(req.address);
    const avail = parseFloat(bal.find(x => x.asset === b.budget.asset)?.available || "0");
    if (avail < parseFloat(b.budget.limit)) { release(); return fail(res, 402, "insufficient_balance", `Escrow ${avail} ${b.budget.asset} < budget ${b.budget.limit}.`); }
  } catch (e) { release(); return fail(res, 502, "escrow_error", e.message); }

  // SSH: install the caller's key, or mint one in-enclave and return it ONCE.
  let keySource = "provided", authorizedKey = (b.sshPublicKey || "").trim(), oneTimePrivateKey = null;
  if (!authorizedKey) {
    try { const kp = generateSshKeypair(`nan:${req.address.slice(0, 10)}`);
          authorizedKey = kp.publicKey; oneTimePrivateKey = kp.privateKey; keySource = "generated"; }
    catch (e) { release(); return fail(res, 500, "keygen_error", "Could not generate an SSH key: " + e.message); }
  }

  const id = rid("dep_");
  let internalPort, sshPort;
  try { ({ internalPort, sshPort } = await spawnContainer({ deploymentId: id, owner: req.address, image,
            command: b.command || [], env: b.env || {}, port: appPort,
            gpu: gpu ? { cardId: gpu.cardId, cardUuid: gpuCards[gpu.cardId]?.uuid || null,
                         vramCapGb: gpu.vramGb, computeShare: gpu.computeShare } : null,
            budget: b.budget, authorizedKey })); }
  catch (e) { release(); return fail(res, 502, "enclave_error", e.message); }

  const now = new Date().toISOString();
  const rec = {
    id, owner: req.address, status: "running",
    image, command: b.command || [],
    resources: gpu ? { vramGb: slice.vramGb, computeShare: round3(slice.computeShare), cardId: gpu.cardId }
                   : { vramGb: 0 },
    network: { port: appPort, protocol: "https", endpoint: `${originOf(req)}/x/${id}` },
    budget: { asset: b.budget.asset, limit: b.budget.limit, spent: "0.00", ratePerSecond: rate.toFixed(7) },
    attestation: { available: true, vmTechnology: "intel-tdx", gpuTechnology: gpu ? "nvidia-cc" : null,
                   href: `/v1/deployments/${id}/attestation` },
    region: "tinfoil", createdAt: now, startedAt: Date.now(), expiresAt: null,
    digest: image.digest || null, rate, _gpu: gpu, _port: internalPort,
    _sshPort: sshPort, _sshKeySource: keySource, _authorizedKey: authorizedKey,
  };
  deployments.set(id, rec);
  // No separate access token: the browser reuses its session Bearer on /x/:id (and the SSH tunnel).
  const out = view(rec);
  if (oneTimePrivateKey) out.ssh.privateKey = oneTimePrivateKey; // shown once; never persisted
  res.status(201).json(out);
});

app.get("/v1/deployments", authed, (req, res) =>
  res.json({ data: [...deployments.values()].filter(d => d.owner === req.address).map(view), cursor: null }));

app.get("/v1/deployments/:id", authed, (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  res.json(view(rec));
});

app.delete("/v1/deployments/:id", authed, async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  try { await stopContainer(rec); } catch {}
  if (rec._gpu) { releaseGpu(rec._gpu); rec._gpu = null; } // return the slice (vram + compute) to the card
  const settled = spentOf(rec); rec.status = "stopping";
  res.json({ id: rec.id, status: "stopping",
             settled:  { asset: rec.budget.asset, amount: settled },
             released: { asset: rec.budget.asset, amount: (parseFloat(rec.budget.limit) - parseFloat(settled)).toFixed(2) } });
});

app.get("/v1/deployments/:id/attestation", authed, async (req, res) => {
  const rec = deployments.get(req.params.id);
  if (!rec || rec.owner !== req.address) return fail(res, 404, "not_found", "No such deployment.");
  try { res.json({ deploymentId: rec.id, generatedAt: new Date().toISOString(), ...(await getMeasurements(rec)), guideUrl: "https://nan.host/#attest" }); }
  catch (e) { fail(res, 502, "attestation_error", e.message); }
});

app.use((_req, res) => fail(res, 404, "not_found", "No such route."));
await initGpu();
await initSshHostKey();

// ---------------------------------------------------------------------------
// SSH TUNNEL — ssh rides the one attested origin as a WebSocket at /x/:id/ssh.
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

server.on("upgrade", async (req, socket, head) => {
  const m = (req.url || "").match(/^\/x\/([^/?]+)\/ssh(?:\?|$)/);
  if (!m) { socket.destroy(); return; }
  const rec  = deployments.get(m[1]);
  const addr = await authUpgrade(req);
  const deny = (line) => { socket.write(`HTTP/1.1 ${line}\r\n\r\n`); socket.destroy(); };
  if (!rec || !rec._sshPort)     return deny("404 Not Found");
  if (!addr)                     return deny("401 Unauthorized");
  if (rec.owner !== addr)        return deny("403 Forbidden");
  if (rec.status !== "running")  return deny("409 Conflict");
  wss.handleUpgrade(req, socket, head, (ws) => {
    // bridge ws <-> sandbox sshd (raw TCP). Binary frames in both directions.
    const tcp = net.connect(rec._sshPort, "127.0.0.1");
    const close = () => { try { ws.close(); } catch {} try { tcp.destroy(); } catch {} };
    tcp.on("connect", () => {
      ws.on("message", (d) => tcp.write(d));
      tcp.on("data", (d) => { if (ws.readyState === ws.OPEN) ws.send(d); });
    });
    ws.on("close", close); ws.on("error", close);
    tcp.on("close", close); tcp.on("error", close);
  });
});

server.listen(PORT, () => console.log(`nan supervisor on :${PORT} · ${GPU_COUNT}×GPU @ ${CARD_VRAM_GB}GB (arbitrary split) · ssh host key ${SSH_HOST_KEY_FP}`));
