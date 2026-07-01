#!/usr/bin/env node
// deploy-nanpay.mjs - compile + deploy contracts/NanPay.sol to Base, print the
// address, and (optionally) write it into tinfoil-config.yml as FORWARDER_ADDRESS.
//
// NanPay is non-custodial: it forwards USDC payer -> payout in the same tx and
// holds nothing. The deployer EOA becomes `owner` (can later setPayout/setOwner).
//
// Deps (run from repo root):  npm i viem solc
//
// Usage:
//   DEPLOYER_PRIVATE_KEY=0x... PAYOUT_ADDRESS=0x...coldwallet \
//     node scripts/deploy-nanpay.mjs                 # -> Base SEPOLIA testnet (default)
//
//   NETWORK=base DEPLOYER_PRIVATE_KEY=0x... PAYOUT_ADDRESS=0x... \
//     node scripts/deploy-nanpay.mjs                 # -> Base MAINNET
//
// On a successful deploy it writes FORWARDER_ADDRESS into tinfoil-config.yml
// automatically (pass --no-write-config to skip).
//
// Env:
//   DEPLOYER_PRIVATE_KEY  required. Pays gas, becomes contract owner.
//   PAYOUT_ADDRESS        cold wallet that receives USDC. If unset, resolves
//                         PAYOUT_ENS (default nan.eth) via ETH_RPC (L1 mainnet).
//   PAYOUT_ENS            ENS name to resolve when PAYOUT_ADDRESS is unset.
//   NETWORK               base-sepolia (default) | base
//   RPC_URL               override the chain RPC.
//   USDC_ADDRESS          override the USDC token address for the network.
//   ETH_RPC               L1 RPC for ENS resolution (default cloudflare-eth.com).
// Flags:
//   --no-write-config     do NOT touch tinfoil-config.yml (default is to write it)
//   --yes                 skip the interactive confirmation (CI)
//   --dry-run             compile + show the plan, do NOT broadcast

import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import readline from "node:readline/promises";
import rlSync from "node:readline";
import { stdin as input, stdout as output } from "node:process";
import solc from "solc";
import {
  createWalletClient, createPublicClient, http, formatEther, isAddress, getAddress,
} from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { base, baseSepolia, mainnet } from "viem/chains";
import { normalize } from "viem/ens";

const HERE = path.dirname(url.fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "..");
const CONTRACT = path.join(REPO, "contracts", "NanPay.sol");
const CONFIG = path.join(REPO, "tinfoil-config.yml");

const args = new Set(process.argv.slice(2));
const DRY_RUN = args.has("--dry-run");
const NO_WRITE_CONFIG = args.has("--no-write-config"); // config is written by default on a successful deploy
const ASSUME_YES = args.has("--yes");

const NETWORKS = {
  "base-sepolia": { chain: baseSepolia, rpc: "https://sepolia.base.org",
                    usdc: "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
                    explorer: "https://sepolia.basescan.org" },
  "base":         { chain: base, rpc: "https://mainnet.base.org",
                    usdc: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    explorer: "https://basescan.org" },
};

function die(msg) { console.error(`\nERROR: ${msg}\n`); process.exit(1); }

// Prompt for a secret with the typed characters hidden (not echoed, not in history).
function promptSecret(query) {
  return new Promise((resolve) => {
    const rl = rlSync.createInterface({ input, output, terminal: true });
    rl._writeToOutput = (s) => { if (!rl._muted) output.write(s); };  // mute keystrokes
    rl.question(query, (ans) => { rl.close(); output.write("\n"); resolve(ans.trim()); });
    rl._muted = true;                                                  // query already printed
  });
}
// Prompt for a normal (non-secret) value.
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
    sources: { "NanPay.sol": { content: source } },
    settings: {
      optimizer: { enabled: true, runs: 200 },
      outputSelection: { "*": { "*": ["abi", "evm.bytecode.object"] } },
    },
  };
  const out = JSON.parse(solc.compile(JSON.stringify(input)));
  const errs = (out.errors || []).filter((e) => e.severity === "error");
  if (errs.length) die("solc:\n" + errs.map((e) => e.formattedMessage).join("\n"));
  const c = out.contracts["NanPay.sol"]["NanPay"];
  return { abi: c.abi, bytecode: "0x" + c.evm.bytecode.object };
}

async function resolvePayout(explicit) {
  explicit = explicit || process.env.PAYOUT_ADDRESS;
  if (explicit) {
    if (!isAddress(explicit)) die(`payout is not a valid address: ${explicit}`);
    return getAddress(explicit);
  }
  const name = process.env.PAYOUT_ENS || "nan.eth";
  const ethRpc = process.env.ETH_RPC || "https://cloudflare-eth.com";
  console.log(`PAYOUT_ADDRESS unset; resolving ${name} via L1 (${ethRpc})...`);
  const l1 = createPublicClient({ chain: mainnet, transport: http(ethRpc) });
  const addr = await l1.getEnsAddress({ name: normalize(name) }).catch((e) => die(`ENS resolve failed: ${e.message}`));
  if (!addr) die(`${name} does not resolve to an address. Set PAYOUT_ADDRESS explicitly.`);
  console.log(`  ${name} -> ${addr}`);
  return getAddress(addr);
}

function writeForwarder(addr) {
  let cfg = fs.readFileSync(CONFIG, "utf8");
  const re = /(-\s*FORWARDER_ADDRESS:\s*)"[^"]*"/;
  if (!re.test(cfg)) die(`could not find FORWARDER_ADDRESS line in ${CONFIG}`);
  cfg = cfg.replace(re, `$1"${addr}"`);
  fs.writeFileSync(CONFIG, cfg);
  console.log(`Wrote FORWARDER_ADDRESS="${addr}" into ${path.relative(REPO, CONFIG)}`);
}

// Pick the network: from $NETWORK if set, else an interactive menu (number or name).
async function chooseNetwork() {
  let n = process.env.NETWORK;
  if (!n && !ASSUME_YES && input.isTTY) {
    const keys = Object.keys(NETWORKS);
    output.write("\nSelect network:\n");
    keys.forEach((k, i) => {
      const tag = k === "base" ? "  (MAINNET - real funds)" : "  (testnet)";
      output.write(`  ${i + 1}) ${k}${tag}\n`);
    });
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
  const usdc = getAddress(process.env.USDC_ADDRESS || net.usdc);

  let pk0 = process.env.DEPLOYER_PRIVATE_KEY;
  if (!pk0) {
    if (ASSUME_YES || !input.isTTY) die("DEPLOYER_PRIVATE_KEY is required (set the env var, or run in a terminal to be prompted)");
    pk0 = await promptSecret("Deployer private key (input hidden, paste and press Enter): ");
  }
  if (!pk0) die("no private key provided");
  const pk = pk0.startsWith("0x") ? pk0 : "0x" + pk0;
  let account; try { account = privateKeyToAccount(pk); } catch { die("that is not a valid private key"); }

  const { abi, bytecode } = compile();
  let payoutIn = process.env.PAYOUT_ADDRESS;
  if (!payoutIn && !ASSUME_YES && input.isTTY)
    payoutIn = await promptText("Payout address (where USDC lands; blank to resolve nan.eth): ");
  const payout = await resolvePayout(payoutIn);

  const pub = createPublicClient({ chain: net.chain, transport: http(rpc) });
  const wallet = createWalletClient({ account, chain: net.chain, transport: http(rpc) });
  const bal = await pub.getBalance({ address: account.address }).catch(() => 0n);

  console.log("\n========================  DEPLOY PLAN  ========================");
  console.log(`  network        ${netName}  (chainId ${net.chain.id})${isMainnet ? "   *** MAINNET / REAL FUNDS ***" : "   (testnet)"}`);
  console.log(`  rpc            ${rpc}`);
  console.log(`  deployer       ${account.address}`);
  console.log(`  deployer ETH   ${formatEther(bal)}`);
  console.log(`  contract       NanPay  (owner = deployer)`);
  console.log(`  usdc  (arg1)   ${usdc}`);
  console.log(`  payout(arg2)   ${payout}   <- USDC lands here`);
  console.log(`  bytecode       ${(bytecode.length / 2 - 1)} bytes`);
  console.log("===============================================================\n");

  if (bal === 0n) die("deployer has 0 ETH on this chain; fund it for gas first.");
  if (DRY_RUN) { console.log("--dry-run: compiled and validated; not broadcasting."); return; }

  if (!ASSUME_YES) {
    const rl = readline.createInterface({ input, output });
    const prompt = isMainnet
      ? `Type "DEPLOY MAINNET" to deploy to Base mainnet with the payout above: `
      : `Deploy to ${netName}? [y/N]: `;
    const ans = (await rl.question(prompt)).trim();
    rl.close();
    const ok = isMainnet ? ans === "DEPLOY MAINNET" : /^y(es)?$/i.test(ans);
    if (!ok) die("aborted by user.");
  }

  console.log("Deploying...");
  const hash = await wallet.deployContract({ abi, bytecode, args: [usdc, payout] });
  console.log(`  tx ${hash}`);
  const rcpt = await pub.waitForTransactionReceipt({ hash });
  if (rcpt.status !== "success" || !rcpt.contractAddress) die(`deploy tx did not succeed (status=${rcpt.status})`);
  const addr = getAddress(rcpt.contractAddress);

  console.log("\n=======================  DEPLOYED  ============================");
  console.log(`  NanPay address   ${addr}`);
  console.log(`  explorer         ${net.explorer}/address/${addr}`);
  console.log(`  set in config    FORWARDER_ADDRESS: "${addr}"`);
  console.log("===============================================================\n");

  if (!NO_WRITE_CONFIG) writeForwarder(addr);
  else console.log("(--no-write-config: skipped tinfoil-config.yml; set FORWARDER_ADDRESS manually)");

  console.log("\nNext:");
  console.log(`  1. Set FORWARDER_ADDRESS in tinfoil-config.yml to ${addr}`);
  console.log("  2. Rebuild+repin the supervisor:  ./scripts/release.sh nan");
  console.log("  3. Confirm the enclave has outbound egress to BASE_RPC so it can watch this contract.");
  if (!isMainnet) console.log("  4. Test the pay flow on testnet, THEN re-run with NETWORK=base for mainnet.");
}

main().catch((e) => die(e.message || String(e)));
