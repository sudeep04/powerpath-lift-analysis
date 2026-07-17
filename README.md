# PowerPath

PowerPath is a Mac-local, mostly-offline video analyzer for the CrossFit barbell
catalog. Film a set on your phone, drop it into the app, and get rep-by-rep form
feedback: bar path, phase boundaries, fault calls, and made/missed per rep. The
differentiator over bar-path-only apps: PowerPath tracks the **bar and your body**
together, so it can explain *why* a path went wrong ("your hips rose before the bar
left the floor") — and its reference is **you**: your best reps become your personal
template per movement, not a generic ideal.

## Quickstart

Prerequisites (macOS):

- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- Node 20+ — `brew install node`

Then:

```sh
./run.sh
```

The script installs dependencies on first run (`uv sync` in `engine/`, `npm install`
in `app/`), starts the analysis API on `127.0.0.1:8400` and the web app on
`localhost:3000`, waits for both to come up, and opens your browser. Ctrl-C tears
both down.

> Open the app at **http://localhost:3000** (not `127.0.0.1:3000`) — Next.js dev
> mode blocks cross-origin dev resources from `127.0.0.1` and hydration silently
> fails.

No pose model installed yet? Run a demo with canned analysis results:

```sh
POWERPATH_FAKE_ENGINE=1 ./run.sh
```

To enable real pose inference (rtmlib + onnxruntime):

```sh
cd engine && uv sync --extra pose        # rtmlib (default backend)
cd engine && uv sync --extra mediapipe   # MediaPipe (alternative backend)
```

The API picks its pose backend from the `POWERPATH_POSE` env var (`rtmlib` by
default; `mediapipe` or `fake` also work).

## CLI

The engine is also usable directly from the command line (run from `engine/`):

```sh
# List the registered movement keys
uv run powerpath movements

# Analyze a video: writes metrics.json, overlay.json, annotated.mp4
# (default output dir: next to the video) and prints a rep table
uv run powerpath analyze ~/lifts/clean.mp4 --movement power_clean --load 80 \
    --height 178 --out /tmp/clean-analysis

# No pose model installed? Use the fake backend (deterministic synthetic pose):
uv run powerpath analyze ~/lifts/clean.mp4 --movement power_clean --load 80 --pose fake

# Freeze bar + landmark series JSON for fixture tests (full-rate pose)
uv run powerpath extract-fixtures ~/lifts/clean.mp4 --movement power_clean \
    --out tests/fixtures/clean
```

`--pose` accepts `fake | rtmlib | mediapipe` (default `rtmlib`, which requires
`uv sync --extra pose`).

Movements registered today (the 5 M1 family representatives + one family-mate):

| Key | Movement |
| --- | --- |
| `power_clean` | Power Clean |
| `power_snatch` | Power Snatch |
| `back_squat` | Back Squat |
| `push_press` | Push Press |
| `deadlift` | Deadlift |
| `hang_power_clean` | Hang Power Clean |

The full 14-movement CrossFit catalog (squat, clean, press/jerk, snatch, deadlift
families + thruster) is designed as registry configs; the remaining family-mates
land in later milestones — each is a config + deltas module, not new engine code.

## Filming guide (The Assignment)

The engine is validated against ground-truth videos you film yourself. For the 5
family representatives — **power clean, back squat, push press, power snatch,
deadlift**:

- **Side-on from a tripod, ~3m away, at hip height.** 60fps, HDR off (or note the
  setting — it affects decoding).
- **Portrait orientation for overhead movements** (push press / jerks / snatch /
  overhead squat) — overhead lockout crops out of landscape at hip height.
  Landscape is fine for squat, clean, and deadlift.
- **Mark the bar with a centered dot or ring on the end cap.** Centered matters:
  sleeves spin during turnover, and an off-center tape strip orbits the bar axis by
  up to ~2.5cm — 2.5x the tracking accuracy gate.
- **Take a test shot per setup** and verify the full range of motion — floor to
  overhead lockout — is in frame before filming the set.
- **Film 2–3 deliberate-fault reps per movement** (intentional early arm bend, cut
  depth, forward drift). These unambiguous positives are what make fault-detection
  validation meaningful.

## Architecture

```
phone video (AirDrop/drag)
        │
  Next.js UI (localhost:3000) ──upload + movement + load──▶ FastAPI (127.0.0.1:8400)
        ▲                                                        │
        │  overlay JSON, metrics,                                ▼
        │  rep data (all via API)                        Streaming engine
        │                                                 decode (PyAV) → pose + bar
   canvas overlay on <video>,                             tracking → calibration →
   rep filmstrip                                          rep segmentation → phase
        │                                                 detection → metrics, fault
   FastAPI owns SQLite ◀── videos, jobs,                  rules, made/missed →
   (UI never touches the DB)   analyses                   overlay JSON + annotated MP4

   (future, M5: Gemini coach notes — metrics JSON only, never video)
```

Analysis is CPU-bound, so it runs in a worker process off the API event loop; the
UI polls the job until it's done. The API binds `127.0.0.1` only — these are
private gym videos, nothing leaves the machine.

## Testing

```sh
cd engine && uv run pytest      # engine suite (no model inference required)
cd app && npx vitest run        # web app unit tests
cd app && npm run e2e           # Playwright end-to-end tests
```

## Status

**Current scope (M1–M2):** the engine pipeline (decode → pose → bar tracking →
calibration → segmentation → phases → metrics/faults → overlay + annotated MP4)
runs end-to-end on uploaded video through the web app, with the 6 movements above
registered. Pose model extras are optional installs; the fake backends
(`--pose fake`, `POWERPATH_FAKE_ENGINE=1`) let everything run without them.

**Deferred to later milestones:** validation of real pose models against the
ground-truth filming set, rep quality scores + personal templates + ghost-rep
overlay (M3), the remaining catalog movements and thruster complex (M3–M4),
Gemini coach notes and progress charts (M5), and live coaching mode (M6).
