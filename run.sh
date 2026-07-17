#!/usr/bin/env bash
# PowerPath launcher: starts the analysis API (FastAPI, 127.0.0.1:8400) and the
# web app (Next.js, localhost:3000), waits for both, opens the browser, and
# tears everything down on Ctrl-C.
#
#   ./run.sh                          # normal run
#   POWERPATH_FAKE_ENGINE=1 ./run.sh  # canned analysis results, no pose model needed
set -euo pipefail
set -m # job control: each background job gets its own process group,
       # so the cleanup trap can kill the whole tree (uv->python, npm->next).

ROOT="$(cd "$(dirname "$0")" && pwd)"
API_URL="http://127.0.0.1:8400/api/movements"
APP_URL="http://localhost:3000" # localhost, NOT 127.0.0.1 -- Next dev blocks
                                # cross-origin dev resources from 127.0.0.1 and
                                # hydration silently fails.
API_PID=""
APP_PID=""

# ---------------------------------------------------------------- preflight --
need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: '$1' is not installed." >&2
    echo "       $2" >&2
    exit 1
  fi
}
need uv "Install it with: brew install uv   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
need node "Install Node 20+ with: brew install node   (or via https://nodejs.org)"
need npm "npm ships with Node -- install Node 20+ with: brew install node"

if [[ ! -d "$ROOT/engine/.venv" ]]; then
  echo "==> engine/.venv not found -- running 'uv sync' (first run only)"
  (cd "$ROOT/engine" && uv sync)
fi
if [[ ! -d "$ROOT/app/node_modules" ]]; then
  echo "==> app/node_modules not found -- running 'npm install' (first run only)"
  (cd "$ROOT/app" && npm install)
fi

if [[ "${POWERPATH_FAKE_ENGINE:-}" == "1" ]]; then
  echo "==> POWERPATH_FAKE_ENGINE=1: the API will return canned analysis results (no pose model needed)"
fi

# ------------------------------------------------------------------ cleanup --
cleanup() {
  trap - EXIT INT TERM
  echo ""
  echo "==> Shutting down PowerPath..."
  # Negative PID = kill the process group (uv/npm children included).
  [[ -n "$API_PID" ]] && kill -TERM -- "-$API_PID" 2>/dev/null || true
  [[ -n "$APP_PID" ]] && kill -TERM -- "-$APP_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  echo "==> Done."
}
on_signal() {
  cleanup
  exit 130
}
trap on_signal INT TERM
trap cleanup EXIT

# ------------------------------------------------------------------- launch --
echo "==> Starting analysis API (uv run powerpath-api) on 127.0.0.1:8400"
(cd "$ROOT/engine" && exec uv run powerpath-api) &
API_PID=$!

echo "==> Starting web app (npm run dev) on localhost:3000"
(cd "$ROOT/app" && exec npm run dev) &
APP_PID=$!

# -------------------------------------------------------------- wait for up --
wait_for() {
  local name="$1" url="$2" pid="$3"
  local deadline=$((SECONDS + 30))
  while ((SECONDS < deadline)); do
    if curl -sf -o /dev/null --max-time 2 "$url"; then
      echo "==> $name is up ($url)"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: $name exited before it came up -- see the log output above." >&2
      return 1
    fi
    sleep 1
  done
  echo "ERROR: $name did not respond at $url within 30s." >&2
  return 1
}

wait_for "API" "$API_URL" "$API_PID"
wait_for "Web app" "$APP_URL" "$APP_PID"

# --------------------------------------------------------------------- open --
echo "==> Opening $APP_URL"
if command -v open >/dev/null 2>&1; then
  open "$APP_URL"
else
  echo "    (no 'open' command -- browse to $APP_URL yourself)"
fi

echo ""
echo "PowerPath is running. Press Ctrl-C to stop both servers."
wait
