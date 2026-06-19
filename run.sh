#!/usr/bin/env bash
# Launch the Sabi web app (offline). Open the printed URL in your browser.
set -euo pipefail
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python -m sabi serve "$@"
