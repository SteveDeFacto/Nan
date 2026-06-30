#!/usr/bin/env bash
# release.sh — build + push every NAN container for linux/amd64, read each pushed
# digest, and repin it in tinfoil-config.yml. Pin by DIGEST (what Tinfoil attests),
# not by tag. Run from anywhere inside the repo.
#
#   ./scripts/release.sh                  # all images
#   ./scripts/release.sh nan vmmanager    # just these
#   DRY_RUN=1 ./scripts/release.sh        # no docker; repin with a fake digest (test)
#
# Auth: set CR_PAT (classic PAT with write:packages) or be logged in to ghcr.io.
# Note: GHCR package VISIBILITY is per-package and sticky. Images already made
# public stay public on re-push; only a brand-new package name needs the one-time
# web-UI publicize (Settings -> Danger Zone -> Change visibility -> Public).
set -euo pipefail

REGISTRY="ghcr.io"
ORG="stevedefacto"
PLATFORM="linux/amd64"          # Tinfoil enclave is x86; guest kernel/qemu are x86
DIGEST_SEP="${DIGEST_SEP:-:}"   # repo uses ':'. If Tinfoil rejects it, set DIGEST_SEP=@
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo dev)}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CONFIG="$REPO_ROOT/tinfoil-config.yml"
cd "$REPO_ROOT"

# image short-name -> build context (Dockerfile is <context>/Dockerfile)
declare -A CONTEXT=(
  [nan-mps]="mps-daemon"
  [nan-worker]="worker"
  [nan]="."
  [nan-vmmanager]="vm"
)
ORDER=(nan-mps nan-worker nan nan-vmmanager)   # deterministic build order

# pick the subset (positional args) or all
TARGETS=("$@"); [ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${ORDER[@]}")

DRY_RUN="${DRY_RUN:-0}"
say(){ printf '\033[1;36m[release]\033[0m %s\n' "$*"; }

# optional login
if [ "$DRY_RUN" != "1" ] && [ -n "${CR_PAT:-}" ]; then
  say "logging in to $REGISTRY as $ORG"
  echo "$CR_PAT" | docker login "$REGISTRY" -u "$ORG" --password-stdin >/dev/null
fi

repin(){ # $1=image name, $2=digest (sha256:...)
  local name="$1" digest="$2"
  local pat="(image: \"${REGISTRY//./\\.}/${ORG}/${name})[:@][^\"]*\""
  local rep="\\1${DIGEST_SEP}${digest}\""
  sed -i -E "s#${pat}#${rep}#" "$CONFIG"
  grep -q "${REGISTRY}/${ORG}/${name}${DIGEST_SEP}${digest}" "$CONFIG" \
    || { echo "ERROR: failed to repin ${name} in $CONFIG"; exit 1; }
}

for name in "${TARGETS[@]}"; do
  ctx="${CONTEXT[$name]:-}"
  [ -z "$ctx" ] && { echo "unknown image: $name (known: ${!CONTEXT[*]})"; exit 1; }
  ref="${REGISTRY}/${ORG}/${name}:${TAG}"

  if [ "$DRY_RUN" = "1" ]; then
    digest="sha256:$(printf '%s' "$name-$TAG" | sha256sum | cut -c1-64)"
    say "DRY_RUN ${name}  (context: ${ctx})  fake digest ${digest}"
  else
    say "build+push ${name}  (context: ${ctx})  -> ${ref}"
    meta="$(mktemp)"
    docker buildx build --platform "$PLATFORM" --push \
      -t "$ref" --metadata-file "$meta" "$ctx"
    digest="$(python3 -c "import json;print(json.load(open('$meta'))['containerimage.digest'])")"
    rm -f "$meta"
    say "  pushed digest ${digest}"
  fi

  repin "$name" "$digest"
  say "  repinned ${name} -> ${DIGEST_SEP}${digest}"
done

# validate the config still parses
python3 -c "import yaml; yaml.safe_load(open('$CONFIG'))" \
  && say "tinfoil-config.yml valid"

say "done. pinned images:"
grep -nE 'image:' "$CONFIG" | sed 's/^/    /'
[ "$DRY_RUN" = "1" ] && say "DRY_RUN: no images were built or pushed; digests are fake."
say "reminder: confirm each package is PUBLIC so Tinfoil can pull it anonymously."
