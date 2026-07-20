#!/usr/bin/env bash
# Snapshot Replay Lab launcher.
#
# This tool imports `leaping` (to build faithful requests). leaping's pinned deps
# can no longer be resolved standalone (the only safe semantic-router, post
# CVE-2026-42208, forces litellm>=1.84 -> openai>=2 / tiktoken 0.12 / tokenizers
# 0.22, which conflict with leaping's pins). The monorepo works only via its
# committed uv.lock. So instead of re-resolving, we run on the monorepo's venv,
# which already has leaping + all core deps; Streamlit/pandas are added additively
# (12 new packages, no changes to existing ones).
set -euo pipefail
cd "$(dirname "$0")"

# Path to the leaping monorepo (override with LEAPING_REPO=/path ./run.sh)
MONO="${LEAPING_REPO:-../leaping}"

if [ ! -d "$MONO/.venv" ]; then
  echo "error: monorepo venv not found at $MONO/.venv" >&2
  echo "Run 'uv sync' in the leaping monorepo first, or set LEAPING_REPO." >&2
  exit 1
fi

if [ ! -x "$MONO/.venv/bin/streamlit" ]; then
  echo "Adding Streamlit + pandas to the monorepo venv (one-time, additive)…"
  uv pip install --python "$MONO/.venv" streamlit pandas
fi

# Load the API keys + Google/Vertex creds that leaping.config validates at import.
set -a
# shellcheck disable=SC1091
source "$MONO/voice/.env"
set +a

exec "$MONO/.venv/bin/streamlit" run app.py "$@"
