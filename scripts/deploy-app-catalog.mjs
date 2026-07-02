#!/usr/bin/env node
// deploy-app-catalog.mjs - compile + deploy contracts/NanAppCatalog.sol to Base,
// print the address, and (optionally) write it into site/index.html as the
// APP_CATALOG_ADDRESS the store reads from.
//
// NanAppCatalog has no constructor args: the deployer EOA becomes `owner` (the
// only address that can flip an app's `verified` flag; can later transferOwnership).
//
// Deps (run from repo root):  npm i viem solc
//
// Usage:
//   DEPLOYER_PRIVATE_KEY=0x... node scripts/deploy-app-catalog.mjs                 # -> Base SEPOLIA (default), auto-wires the site
//   NETWORK=base DEPLOYER_PRIVATE_KEY=0x... node scripts/deploy-app-catalog.mjs    # -> Base MAINNET, auto-wires the site
//
// On a successful deploy it writes APP_CATALOG_ADDRESS / _CHAIN / _RPC into
// site/index.html automatically (pass --no-write-config to skip).
//
// Env:
//   DEPLOYER_PRIVATE_KEY  required. Pays gas, becomes contract owner.
//   NETWORK               base-sepolia (default) | base
//   RPC_URL               override the chain RPC.
// Flags:
//   --no-write-config     do NOT touch site/index.html (default is to wire it)
//   --yes                 skip the interactive confirmation (CI)
//   --dry-run             compile + show the plan, do NOT broadcast

import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import readline from "node:readline/promises";
import rlSync from "node:readline";
import { stdin as input, stdout as output } from "node:process";
import solc from "solc";
import { createWalletClient, createPublicClient, http, formatEther, getAddress } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { base, baseSepolia } from "viem/chains";

const HERE = path.dirname(url.fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..");
const CONTRACT = path.join(REPO, "contracts", "NanAppCatalog.sol");
const ABI_OUT = path.join(REPO, "contracts", "NanAppCatalog.abi.json");
const SITE = path.join(REPO, "site", "index.html");
const CONFIG = path.join(REPO, "tinfoil-config.yml");

const args = new Set(process.argv.slice(2));
const DRY_RUN = args.has("--dry-run");
const NO_WRITE_CONFIG = args.has("--no-write-config"); // config is written by default on a successful deploy
const ASSUME_YES = args.has("--yes");

const NETWORKS = {
  "base-sepolia": { chain: baseSepolia, rpc: "https://sepolia.base.org", explorer: "https://sepolia.basescan.org" },
  "base":         { chain: base,        rpc: "https://mainnet.base.org",  explorer: "https://basescan.org" },
};

function die(msg) { console.error(`\nERROR: ${msg}\n`); process.exit(1); }

function promptSecret(query) {
  return new Promise((resolve) => {
    const rl = rlSync.createInterface({ input, output, terminal: true });
    rl._writeToOutput = (s) => { if (!rl._muted) output.write(s); };
    rl.question(query, (ans) => { rl.close(); output.write("\n"); resolve(ans.trim()); });
    rl._muted = true;
  });
}
async function promptText(query) {
  const rl = readline.createInterface({ input, output });
  const ans = (await rl.question(query)).trim();
  rl.close();
  return ans;
}

function compile() {
  const source = fs.readFileSync(CONTRACT, "utf8");
  const input = {
    language: "Solidity",
    sources: { "NanAppCatalog.sol": { content: source } },
    // viaIR: publishVersion's 7 params (6 dynamic) overflow the legacy codegen's
    // calldata decoder ("stack too deep"); the IR pipeline spills to memory.
    settings: { optimizer: { enabled: true, runs: 200 }, viaIR: true, outputSelection: { "*": { "*": ["abi", "evm.bytecode.object"] } } },
  };
  const out = JSON.parse(solc.compile(JSON.stringify(input)));
  const errs = (out.errors || []).filter((e) => e.severity === "error");
  if (errs.length) die("solc:\n" + errs.map((e) => e.formattedMessage).join("\n"));
  const c = out.contracts["NanAppCatalog.sol"]["NanAppCatalog"];
  // keep the checked-in ABI in lockstep with what we deploy
  fs.writeFileSync(ABI_OUT, JSON.stringify(c.abi, null, 2) + "\n");
  return { abi: c.abi, bytecode: "0x" + c.evm.bytecode.object };
}

// Point the site at the deployment: address + the chain/RPC it lives on, so a
// testnet deploy yields a working site and a mainnet deploy flips it back.
function writeSiteConfig(addr, chainId, rpc) {
  let html = fs.readFileSync(SITE, "utf8");
  const subs = [
    [/(const APP_CATALOG_ADDRESS\s*=\s*)"[^"]*"/, `$1"${addr}"`,      `APP_CATALOG_ADDRESS="${addr}"`],
    [/(const APP_CATALOG_CHAIN\s*=\s*)\d+/,       `$1${chainId}`,      `APP_CATALOG_CHAIN=${chainId}`],
    [/(const APP_CATALOG_RPC\s*=\s*)"[^"]*"/,     `$1"${rpc}"`,        `APP_CATALOG_RPC="${rpc}"`],
  ];
  for (const [re] of subs) if (!re.test(html)) die(`could not find ${re} in ${SITE}`);
  for (const [re, rep] of subs) html = html.replace(re, rep);
  fs.writeFileSync(SITE, html);
  console.log(`Wrote ${subs.map((s) => s[2]).join(", ")} into ${path.relative(REPO, SITE)}`);
}

// Point the supervisor at the same deployment: it reads cidStatus() at deploy
// time and refuses ipfs:// apps the catalog owner hasn't Approved. Mirrors
// deploy-nanpay.mjs's FORWARDER_ADDRESS write.
function writeSupervisorConfig(addr, chainId) {
  let cfg = fs.readFileSync(CONFIG, "utf8");
  const re = /(-\s*APP_CATALOG_ADDRESS:\s*)"[^"]*"/;
  if (!re.test(cfg)) die(`could not find APP_CATALOG_ADDRESS line in ${CONFIG}`);
  cfg = cfg.replace(re, `$1"${addr}"`);
  fs.writeFileSync(CONFIG, cfg);
  console.log(`Wrote APP_CATALOG_ADDRESS="${addr}" into ${path.relative(REPO, CONFIG)}`);
  const chainRe = /-\s*CHAIN_ID:\s*"(\d+)"/.exec(cfg);
  if (chainRe && Number(chainRe[1]) !== chainId)
    console.warn(`WARNING: tinfoil-config.yml has CHAIN_ID ${chainRe[1]} but the catalog was deployed to chain ${chainId} — the supervisor reads the catalog over BASE_RPC, so approval checks will fail until they match.`);
}

async function chooseNetwork() {
  let n = process.env.NETWORK;
  if (!n && !ASSUME_YES && input.isTTY) {
    const keys = Object.keys(NETWORKS);
    output.write("\nSelect network:\n");
    keys.forEach((k, i) => output.write(`  ${i + 1}) ${k}${k === "base" ? "  (MAINNET - real funds)" : "  (testnet)"}\n`));
    const ans = await promptText(`Enter number or name [1=${keys[0]}]: `);
    if (!ans) n = keys[0];
    else if (/^\d+$/.test(ans)) n = keys[parseInt(ans, 10) - 1];
    else n = ans;
  }
  return (n || "base-sepolia").toLowerCase();
}

async function main() {
  const netName = await chooseNetwork();
  const net = NETWORKS[netName];
  if (!net) die(`unknown network "${netName}" (valid: ${Object.keys(NETWORKS).join(", ")})`);
  const isMainnet = netName === "base";
  const rpc = process.env.RPC_URL || net.rpc;

  let pk0 = process.env.DEPLOYER_PRIVATE_KEY;
  if (!pk0) {
    if (ASSUME_YES || !input.isTTY) die("DEPLOYER_PRIVATE_KEY is required (set the env var, or run in a terminal to be prompted)");
    pk0 = await promptSecret("Deployer private key (input hidden, paste and press Enter): ");
  }
  if (!pk0) die("no private key provided");
  const pk = pk0.startsWith("0x") ? pk0 : "0x" + pk0;
  let account; try { account = privateKeyToAccount(pk); } catch { die("that is not a valid private key"); }

  const { abi, bytecode } = compile();
  const pub = createPublicClient({ chain: net.chain, transport: http(rpc) });
  const wallet = createWalletClient({ account, chain: net.chain, transport: http(rpc) });
  const bal = await pub.getBalance({ address: account.address }).catch(() => 0n);

  console.log("\n========================  DEPLOY PLAN  ========================");
  console.log(`  network        ${netName}  (chainId ${net.chain.id})${isMainnet ? "   *** MAINNET / REAL FUNDS ***" : "   (testnet)"}`);
  console.log(`  rpc            ${rpc}`);
  console.log(`  deployer       ${account.address}`);
  console.log(`  deployer ETH   ${formatEther(bal)}`);
  console.log(`  contract       NanAppCatalog  (owner = deployer; no constructor args)`);
  console.log(`  bytecode       ${(bytecode.length / 2 - 1)} bytes`);
  console.log("===============================================================\n");

  if (bal === 0n) die("deployer has 0 ETH on this chain; fund it for gas first.");
  if (DRY_RUN) { console.log("--dry-run: compiled and validated; not broadcasting."); return; }

  if (!ASSUME_YES) {
    const rl = readline.createInterface({ input, output });
    const prompt = isMainnet ? `Type "DEPLOY MAINNET" to deploy to Base mainnet: ` : `Deploy to ${netName}? [y/N]: `;
    const ans = (await rl.question(prompt)).trim();
    rl.close();
    const ok = isMainnet ? ans === "DEPLOY MAINNET" : /^y(es)?$/i.test(ans);
    if (!ok) die("aborted by user.");
  }

  console.log("Deploying...");
  const hash = await wallet.deployContract({ abi, bytecode, args: [] });
  console.log(`  tx ${hash}`);
  const rcpt = await pub.waitForTransactionReceipt({ hash });
  if (rcpt.status !== "success" || !rcpt.contractAddress) die(`deploy tx did not succeed (status=${rcpt.status})`);
  const addr = getAddress(rcpt.contractAddress);

  console.log("\n=======================  DEPLOYED  ============================");
  console.log(`  NanAppCatalog     ${addr}`);
  console.log(`  explorer          ${net.explorer}/address/${addr}`);
  console.log(`  set in site        const APP_CATALOG_ADDRESS = "${addr}"`);
  console.log(`  set in site        const APP_CATALOG_CHAIN   = ${net.chain.id}`);
  console.log("===============================================================\n");

  if (!NO_WRITE_CONFIG) { writeSiteConfig(addr, net.chain.id, rpc); writeSupervisorConfig(addr, net.chain.id); }
  else console.log("(--no-write-config: skipped wiring site/index.html + tinfoil-config.yml; set APP_CATALOG_ADDRESS manually)");

  console.log("\nNext:");
  console.log(`  1. site/index.html + tinfoil-config.yml now point at ${addr} on chain ${net.chain.id}.`);
  console.log("  2. Redeploy the site:  cd site && ./deploy.sh");
  console.log("  3. Rebuild+repin the supervisor so the enclave enforces approval:  ./scripts/release.sh nan");
  console.log("  4. Approve versions from the Apps tab with the deployer wallet (it is the catalog owner).");
  if (!isMainnet) console.log("  5. Test publish/approve/deploy on testnet, THEN re-run with NETWORK=base for mainnet.");
}

main().catch((e) => die(e.message || String(e)));
