# NAN on-chain discovery (NanRegistry)

Transparent, gateway-free enclave discovery. The registry is the on-chain source
of truth for *which enclaves exist, where, and what code they claim*. Callers
read it from any RPC and connect to an enclave **directly**, verifying its live
attestation with Tinfoil's SecureClient. There is no trusted middleman: the
contract publishes claims, attestation (at connect time) gates trust.

```
caller --(eth_call)--> NanRegistry            "enclave at URL, repo=org/repo"
caller --(/availability)--> each enclave      live free share (off-chain)
caller --(SecureClient(URL, repo))--> enclave verifies SEV-SNP/TDX + Sigstore, pins TLS
```

## Files
- `contracts/NanRegistry.sol` — the registry (open, no deps, ~120 lines).
- `contracts/NanRegistry.abi.json` — ABI for JS callers.
- supervisor self-registration — built into `supervisor.js` (`registerOnChain`).
- `scripts/nan-discover.mjs` — caller-side read + availability aggregation + pick.

## 1. Deploy the contract to Base (chain 8453)

No constructor args. Any toolchain works; quickest is Remix or Foundry.

Foundry:
```bash
forge create contracts/NanRegistry.sol:NanRegistry \
  --rpc-url https://mainnet.base.org \
  --private-key $DEPLOYER_PK \
  --broadcast
```
Note the deployed address — that's `REGISTRY_ADDRESS` everywhere below. (Test on
Base Sepolia first with its RPC + a faucet if you want a dry run.)

## 2. Give each enclave a registry config

The enclave self-registers on boot when `REGISTRY_ENABLED` is set. Add to the
supervisor container's env in `tinfoil-config.yml`:

```yaml
env:
  REGISTRY_ENABLED: "1"
  REGISTRY_ADDRESS: "0x<deployed registry address>"
  ENCLAVE_ENDPOINT: "https://enclave1.nan.containers.tinfoil.dev"  # this enclave's own URL
  ENCLAVE_REPO:     "SteveDeFacto/Nan"      # what callers attest against (Sigstore-measured)
  # REGISTRY_HEARTBEAT_SEC: "900"           # optional, default 15 min
```

And the operator key as an **enclave secret** (not plaintext env):
```
REGISTRY_PRIVATE_KEY = 0x...   # an EOA that owns this enclave's registry entry
```

> The key needs a little ETH on **Base** for gas: register is one tx at boot,
> then one cheap heartbeat every 15 min (~a few cents/month on Base). Fund the
> address before first boot or registration silently fails (non-fatal — the
> enclave still serves, it just won't advertise until it can pay gas).

On boot you'll see: `[registry] registered https://... repo=... id=0x... tx=0x...`
then periodic `[registry] heartbeat tx=0x...`.

`ENCLAVE_MEASUREMENT` is optional (a cross-check digest); leave unset (0x0) and
let the live attestation be authoritative, which it is regardless.

## 3. Callers discover + connect

Read the registry and aggregate availability (no gateway, runs anywhere):
```bash
REGISTRY_ADDRESS=0x... BASE_RPC=https://mainnet.base.org node scripts/nan-discover.mjs 0.25
```
Prints aggregate free capacity across all live enclaves and the best one for a
0.25 share, ending with the endpoint+repo to hand to SecureClient.

Then connect with attestation (the part that actually gates trust):
```js
// JS/TS, browser or server
import { SecureClient } from "tinfoil";
const c = new SecureClient({ baseURL: chosen.endpoint /*, repo handling per tinfoil-js*/ });
await c.ready();                       // verifies SEV-SNP/TDX + Sigstore, pins TLS
await c.fetch("/availability");        // now provably talking to attested code
await c.fetch(`/x/${deploymentId}/run`, { method: "POST", body });
```
```go
// Go
import "github.com/tinfoilsh/verifier/client"
tc := client.NewSecureClient(chosen.endpoint, chosen.repo)
resp, _ := tc.Get("/availability", nil)
```

## Trust boundaries (what this does and does not do)
- **Discovery is trustless**: the enclave list is public chain state; no operator
  sits in the request path. Picking the wrong enclave costs placement, not safety.
- **Trust is gated by attestation at connect**, done by the caller's SecureClient
  — not by registration. A malicious operator can register an enclave, but its
  live quote won't match a NAN-good `repo`/measurement, so callers reject it.
- **Liveness is advisory**: heartbeats set `lastSeen`; readers drop entries
  staler than their window (the helper uses 1h).
- **Open registration for now.** Sybil resistance via stake-to-register +
  slashing is a future add (see the note at the bottom of the .sol); it isn't
  needed for correctness because attestation, not registration, gates trust.

## Verify on first deploy
After one enclave registers:
```bash
node scripts/nan-discover.mjs        # should list it with live availability
```
You should see it in `all[]` with a real `maxShare`, and `chosen` set. That's the
full loop: chain registry -> live availability -> a pick, with nothing trusted
in the middle.

---

# NAN app store (NanAppCatalog)

A second, independent on-chain contract: the public catalog of **Wasm apps**
users can browse and publish. Where `NanRegistry` answers "which enclaves exist,"
`NanAppCatalog` answers "which apps exist and where their code lives." It backs
the **Apps** tab on the site.

```
browser --(eth_call getAppsPage)------> NanAppCatalog   apps: {appId,slug,name,desc,publisher,versionCount,active}
browser --(eth_call getVersionsPage)--> NanAppCatalog   per app: {cid,version,memMb,verified,yanked}
browser --(fetch CID)-----------------> any IPFS peer   the .wasm bytes (hash == CID, verify yourself)
browser --(publishVersion tx)---------> NanAppCatalog   one Base tx; publisher = msg.sender
```

## Two catalogs, on purpose

- **Attested, baked-in catalog** (`wasm/apps/catalog.json`) — curated apps compiled
  into the measured wasm-manager image. What the enclave runs today is exactly what
  attestation measured. Deploys reference these by id (e.g. `hello`).
- **On-chain community catalog** (`NanAppCatalog`) — open discovery. Anyone
  publishes; each entry is a `wasi:http` component addressed by its **IPFS CID**.
  This is *discovery*, not custody or attestation: the catalog stores the CID (a
  hash of the exact wasm), never the bytes, so a caller fetches from any IPFS peer
  and verifies the bytes match the CID independently.

**Run-by-CID (implemented).** Uploaded apps deploy, not just browse. `image.reference`
accepts:
- a baked-in catalog id (`hello`),
- `ipfs://<cid>`, or
- a human-friendly `slug:version` (or `<publisher>/slug:version` to disambiguate) —
  the **browser** resolves that against the on-chain catalog to the app's CID (unique
  because version labels are unique per app) and sends `ipfs://<cid>`.

The enclave's wasm-manager then fetches the CID from `IPFS_GATEWAY` as a **CAR**,
verifies every block hashes to its CID and reassembles the file rooted at the
requested CID (`wasm/ipfs_fetch.py`), rejects anything that isn't a wasi:http
component, caches it under `APPS_DIR`, and runs it. A tampering gateway fails the
hash check, so the operator's own gateway is fine to use. The verified CID is folded
into the attestation (`getMeasurements().app.cid`), so "what ran = this exact CID"
holds. The store card's **Use in Deploy** sets the friendly `slug:version`.

Manager env: `IPFS_GATEWAY` (default `https://ipfs.nan.host`), `WASM_MAX_BYTES`
(default 256 MiB), `IPFS_FETCH_TIMEOUT`.

## Versioning & trust model
- An **app** is `appId = keccak256(publisher, slug)` — a slug in the publisher's own
  namespace. Because the appId embeds `msg.sender`, only you can ever write to your
  app; lineage ownership is structural, not a spoofable check, and slugs can't be
  squatted across publishers.
- `publishVersion(slug, …)` appends an **immutable Version** (its own CID, label,
  memMb), creating the app on first use. Versions are append-only history; you don't
  edit a release, you publish a new one. `editApp` changes display metadata;
  `setActive` delists the whole app; `yankVersion` pulls a bad release (kept for
  history, hidden by readers).
- **Global CID uniqueness**: a wasm artifact is listed at most once across the whole
  catalog, so a CID maps unambiguously to one app/version.
- `verified` is an OPTIONAL owner-curated signal, set **per version** (you verify a
  specific CID; a new release starts unverified and must be re-checked). It does
  **not** gate execution — the CID does. The site can filter to verified.

## Deploy the contract to Base (chain 8453)

No constructor args; the deployer EOA becomes `owner` (can flip `verified`, can
`transferOwnership`). A viem+solc script mirrors `deploy-nanpay.mjs` and **wires the
site automatically** on success:

```bash
# Base Sepolia dry run (compile + plan, no broadcast):
DEPLOYER_PRIVATE_KEY=0x... node scripts/deploy-app-catalog.mjs --dry-run --yes

# Base Sepolia deploy (auto-wires site to chain 84532 + the sepolia RPC):
DEPLOYER_PRIVATE_KEY=0x... node scripts/deploy-app-catalog.mjs

# Base MAINNET deploy (auto-wires site back to chain 8453 + mainnet RPC):
NETWORK=base DEPLOYER_PRIVATE_KEY=0x... node scripts/deploy-app-catalog.mjs
```

On a successful deploy it rewrites `APP_CATALOG_ADDRESS`, `APP_CATALOG_CHAIN`, and
`APP_CATALOG_RPC` in `site/index.html` to match the network you deployed to (pass
`--no-write-config` to skip). It also re-emits `contracts/NanAppCatalog.abi.json`
from source on every run so the checked-in ABI can't drift from what's deployed.

> **Deploy once.** Re-running deploys a *fresh, empty* contract at a new address
> (and, by default, repoints the site at it). Contracts are immutable — schema/logic
> changes mean a new deploy, so make them before apps are published.

## Wire the site (usually automatic)

The deploy script sets these for you; they only need hand-editing for IPFS or a
custom RPC. Five constants near the top of the page's script:

```js
const APP_CATALOG_ADDRESS = "0x…";                    // written by the deploy script
const APP_CATALOG_CHAIN   = 8453;                     // written by the deploy script (84532 = Base Sepolia)
const APP_CATALOG_RPC     = "https://mainnet.base.org"; // written by the deploy script (must match the chain)
const IPFS_UPLOAD_URL     = "https://ipfs.nan.host/add-wasm"; // validating upload gateway; empty => paste-a-CID
const IPFS_GATEWAY        = "https://ipfs.io/ipfs/";    // where "fetch .wasm" links resolve
```

**Uploads go through a validating gateway, not Kubo directly.** Because the browser's
checks are bypassable, the raw Kubo `/api/v0/add` is **not** exposed. Instead:

```
browser → Caddy (/add-wasm, size cap) → scripts/ipfs-add-gateway.py (127.0.0.1:5051) → Kubo (127.0.0.1:5001)
```

The gateway (`scripts/ipfs-add-gateway.py`, stdlib Python, runs on the VM) re-enforces
size + the wasm/component preamble server-side, optionally runs `wasm-tools validate`
(set `WASM_TOOLS`), then adds+pins with hardcoded params and returns `{cid}`. Set
`IPFS_UPLOAD_URL` to that gateway (empty => users paste a CID they pinned themselves).

**Defense in depth, by layer:**
- Browser (`validateWasm`): extension, size (`MAX_WASM_MB`, default 256), magic `\0asm`
  + component layer field. UX only — fast feedback, catches honest mistakes.
- Caddy `request_body { max_size 257MiB }` on `/add-wasm`: the enforceable size ceiling.
- Gateway: the enforceable **content** gate for what gets pinned to *your node*.
- wasm-manager at deploy (`wasmtime serve`): the authoritative "is it a runnable
  wasi:http component" check.

**What server-side upload validation still can't do:** the catalog contract is
permissionless — anyone can pin bytes elsewhere and call `publishVersion` with that
CID directly, and the contract can't inspect bytes. So bad *listings* remain possible;
they just fail to deploy. To keep the *store view* clean you'd add an off-chain
validating indexer (fetch each listed CID, validate, hide the ones that fail) or lean
on the owner `verified` flag as the trust signal.

The browser ABI codec is hand-rolled (no web3 lib on the page, matching the pay
flow): a generic `encCall` and a schema-driven `decodeStructArray`. Every encoder
plus the `App[]`/`Version[]` decoders were verified byte-for-byte against viem
before shipping.

## Verify on first deploy
Deploy to Base Sepolia (the script wires the site to chain 84532 + the sepolia RPC),
redeploy the site, open the **Apps** tab, publish a slug at `1.0.0`, publish the same
slug again at `1.1.0`, and confirm one card shows both in the version selector. Then
re-deploy to mainnet with `NETWORK=base`.
