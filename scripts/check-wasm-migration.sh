#!/usr/bin/env bash
# Verify the repo is fully migrated to the Wasm backend and flag anything still
# on the old runsc/vm backend or anything that silently blocks a deploy.
# Run from anywhere inside the repo.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

pass=0; fail=0
ok(){   printf '  \033[32m OK \033[0m %s\n' "$1"; pass=$((pass+1)); }
bad(){  printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
warn(){ printf '  \033[33mWARN\033[0m %s\n' "$1"; }
sec(){  printf '\n== %s ==\n' "$1"; }

sec "wasm backend files present"
for f in wasm/wasm_manager.py wasm/Dockerfile.wasm wasm/apps/catalog.json wasm/apps/hello.wasm; do
  [ -f "$f" ] && ok "$f" || bad "$f MISSING"
done
if [ -f wasm/apps/hello.wasm ]; then
  magic=$(head -c4 wasm/apps/hello.wasm 2>/dev/null | od -An -tx1 | tr -d ' \n')
  [ "$magic" = "0061736d" ] && ok "hello.wasm has valid wasm magic" \
    || bad "hello.wasm is not a wasm module (magic=$magic; empty or wrong file?)"
fi
if [ -f wasm/apps/catalog.json ]; then
  if command -v python3 >/dev/null && python3 -c "import json,sys;json.load(open('wasm/apps/catalog.json'))" 2>/dev/null; then
    grep -q '"hello"' wasm/apps/catalog.json && ok "catalog.json valid + lists hello" || warn "catalog.json valid but no hello entry"
  else bad "catalog.json is not valid JSON"; fi
fi

sec "tinfoil-config.yml"
C=tinfoil-config.yml
if [ -f "$C" ]; then
  grep -q 'name: *"*wasm-manager' "$C" && ok "has wasm-manager container" || bad "no wasm-manager container (config not updated)"
  grep -q 'nan-wasm-manager' "$C"      && ok "references nan-wasm-manager image" || bad "no nan-wasm-manager image ref"
  if grep -Eq 'runsc-manager|nan-runsc-manager' "$C"; then bad "STILL references runsc-manager (stale config)"; else ok "no runsc references"; fi
  if grep -vE '^[[:space:]]*#' "$C" | grep -Eq 'privileged: *true'; then warn "privileged:true still present (Tinfoil ignores it; harmless but stale)"; else ok "no privileged flag"; fi
  imgline=$(grep 'nan-wasm-manager' "$C" | head -1)
  if   echo "$imgline" | grep -q '@sha256:';   then ok "wasm-manager image pinned (@sha256:)"
  elif echo "$imgline" | grep -q ':placeholder'; then bad "wasm-manager still :placeholder -> run scripts/release.sh nan-wasm-manager"
  else warn "could not determine wasm-manager image pin: $imgline"; fi
else bad "$C MISSING"; fi

sec "scripts/release.sh"
R=scripts/release.sh
if [ -f "$R" ]; then
  grep -q 'nan-wasm-manager' "$R"     && ok "builds nan-wasm-manager" || bad "release.sh does not build nan-wasm-manager"
  grep -q 'wasm/Dockerfile.wasm' "$R" && ok "maps wasm/Dockerfile.wasm" || bad "release.sh has no wasm/Dockerfile.wasm mapping"
  if grep -Eq 'nan-runsc-manager|vm/Dockerfile.runsc' "$R"; then bad "STILL references runsc (stale release.sh)"; else ok "no runsc references"; fi
else bad "$R MISSING"; fi

sec "wasm_manager.py freshness"
M=wasm/wasm_manager.py
if [ -f "$M" ]; then
  grep -q '"runtime": "wasmtime"' "$M" && ok "reports runtime=wasmtime" || bad "missing wasmtime marker (stale/old manager file)"
  grep -q -- '-Scli' "$M"              && ok "has -Scli -Shttp serve flags (latest)" || warn "missing -Scli flag (older manager version)"
  grep -q 'serve_available' "$M"       && ok "has /debug/env serve check" || warn "older /debug/env"
fi

sec "git / deploy state"
if git rev-parse --git-dir >/dev/null 2>&1; then
  dirty=$(git status --porcelain -- wasm tinfoil-config.yml scripts/release.sh 2>/dev/null)
  if [ -n "$dirty" ]; then
    bad "uncommitted changes (Tinfoil only deploys a committed + attested tag):"
    echo "$dirty" | sed 's/^/         /'
  else ok "wasm/config/release changes are committed"; fi
  if git rev-parse '@{u}' >/dev/null 2>&1; then
    ahead=$(git rev-list --count '@{u}..HEAD' 2>/dev/null || echo '?')
    [ "$ahead" = "0" ] && ok "HEAD is pushed to upstream" || warn "HEAD is $ahead commit(s) ahead of upstream (push a tag so Build-and-Attest runs)"
  else warn "no upstream tracking branch set"; fi
else warn "not a git repo here; cannot check commit/push state"; fi

sec "summary: $pass ok, $fail problem(s)"
if [ "$fail" -eq 0 ]; then
  cat <<'EOF'
Local files look fully migrated to the Wasm backend.
If the live enclave STILL reports runsc at localhost:8091/debug/env, the
remaining step is the deploy chain, not the files:
  1) git commit + push a new tag
  2) let the Build-and-Attest workflow go green
  3) relaunch the enclave onto that tag
Until you relaunch, the old attested release keeps running.
EOF
else
  echo "Fix the FAILs above, then: release.sh -> commit -> push tag -> relaunch enclave."
fi
exit "$fail"
