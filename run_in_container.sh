#!/usr/bin/env bash
set -euo pipefail

PERSIST_BASE=/lustre/fsw/portfolios/coreai/projects/coreai_dlcompiler_agent/users/mfrisella
REPO=$PERSIST_BASE/piper
CODEX_PERSIST=$PERSIST_BASE/.codex

mkdir -p "$CODEX_PERSIST"

# Recreate container-side ~/.codex every time the container starts
rm -rf "$HOME/.codex"
ln -sfn "$CODEX_PERSIST" "$HOME/.codex"

cd "$REPO"

exec bash -i

