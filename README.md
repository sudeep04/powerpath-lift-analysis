# PowerPath

**Barbell lift video analysis that tracks the bar *and* your body — locally, on your Mac.**

Film a set on your phone, drop it into the app, and get rep-by-rep form feedback:
bar path, phase boundaries, fault calls, and made/missed per rep — with an
annotated video to scrub through. Everything runs on-device; your gym videos
never leave your machine.

Two things set it apart from bar-path-only apps (WL Analysis, Metric VBT, …):

- **Bar + body together.** Tracking the skeleton alongside the barbell means the
  app can explain *why* a path went wrong — "your hips rose before the bar left
  the floor" — not just show you a squiggle.
- **The reference is you** *(roadmap)*. Your best reps become your personal
  template per movement, so feedback is "3cm further from your hip than your
  best cleans," not a generic ideal.

## How it works

```
phone video (AirDrop/drag)
        │
  Next.js UI (localhost:3000) ──upload + movement + load──▶ FastAPI (127.0.0.1:8400)
        ▲                                                        │
        │  overlay JSON, metrics,                                ▼
        │  rep data (all via API)                        Streaming engine
        │                                                 decode (PyAV) → pose + bar
   canvas overlay on <video>,                             tracking → calibration →
   rep filmstrip, metrics panel                           rep segmentation → phase
        │                                                 detection → metrics, fault
     FastAPI owns SQLite ◀── videos, jobs,                rules, made/missed, scores →
     (UI never touches the DB)   analyses                 overlay JSON + annotated MP4
```

Nice tricks under the hood:

- **Plate-based calibration**: full-size Olympic plates are a standardized 450mm,
  so detecting the plate gives a pixel→cm scale for free — with a bar-marker
  cross-check so a wrong scale is never silently used.
- **Prominence-based rep segmentation** (`scipy.signal.find_peaks` on the bar
  trajectory): robust to catch dips, walkout jitter, and dropped bars, where
  velocity-threshold state machines fall apart.
- **Streaming single-pass pipeline**: video frames are never buffered; a 30s/60fps
  clip analyzes in well under a minute with ~tens of MB of memory.
- Analysis runs in a worker process off the API event loop (survives hard crashes
  in native code); the UI polls job progress. The API binds `127.0.0.1` only.

## Quickstart

Prerequisites (macOS): [uv](https://docs.astral.sh/uv/) (`brew install uv`) and
Node 20+ (`brew install node`).

```sh
./run.sh
```

First run installs dependencies (`uv sync` in `engine/`, `npm install` in `app/`),
then starts the API on `127.0.0.1:8400` and the web app on `localhost:3000`, and
opens your browser. Ctrl-C tears both down.

> If port 3000 is taken, Next.js falls back to 3001+ — `run.sh` detects the port
> actually serving PowerPath and opens that. Always browse via **localhost**, not
> `127.0.0.1` (Next dev mode blocks cross-origin dev assets from `127.0.0.1`).

### No pose models? Try it anyway

Real pose inference is an optional install. Two ways to explore without it:

```sh
# 1) Demo the full UI with canned analysis results (any video upload works):
POWERPATH_FAKE_ENGINE=1 ./run.sh

# 2) Generate synthetic sample lifts and run the REAL pipeline on them
#    (real decode → bar tracking → calibration → segmentation → scoring;
#    pose is scripted from the synthetic body):
cd engine && uv run python tests/make_samples.py
open ../samples/power_clean/annotated.mp4   # bar path + skeleton drawn on
```

The power-clean sample yields 5 detected reps, all made, with real phases and
scores. (The back-squat sample intentionally shows 5 reps / 0 made: the synthetic
test body doesn't bend its knees, and squat "made" requires below-90° depth —
real footage doesn't have that limitation.)

### Enable real pose inference

```sh
cd engine && uv sync --extra pose        # rtmlib + onnxruntime (default backend)
cd engine && uv sync --extra mediapipe   # MediaPipe (alternative backend)
```

The API picks its backend from `POWERPATH_POSE` (`rtmlib` default; `mediapipe`,
`fake` also work).

## CLI

The engine works standalone (run from `engine/`):

```sh
uv run powerpath movements                 # list registered movement keys

uv run powerpath analyze ~/lifts/clean.mp4 --movement power_clean --load 80 \
    --height 178 --out /tmp/clean-analysis # → metrics.json, overlay.json,
                                           #   annotated.mp4 + a rep table

uv run powerpath extract-fixtures ~/lifts/clean.mp4 --movement power_clean \
    --out tests/fixtures/clean             # freeze series JSON for tests
```

`--pose` accepts `fake | rtmlib | mediapipe` (default `rtmlib`).

## Movements

Registered today (the 5 family representatives + one family-mate):

| Key | Movement | Family |
| --- | --- | --- |
| `power_clean` | Power Clean | clean |
| `hang_power_clean` | Hang Power Clean | clean |
| `power_snatch` | Power Snatch | snatch |
| `back_squat` | Back Squat | squat |
| `push_press` | Push Press | press |
| `deadlift` | Deadlift | hinge |

The full 14-movement CrossFit barbell catalog (front/overhead squat, squat clean,
push/split jerk, squat/hang snatch, thruster, …) is designed as registry configs:
each remaining movement is a config-and-deltas module, not new engine code.

## Filming guide

The engine is validated against ground-truth videos. To film your own:

- **Side-on from a tripod, ~3m away, at hip height.** 60fps, HDR off.
- **Portrait orientation for overhead movements** (press/jerk/snatch) — overhead
  lockout crops out of landscape at hip height. Landscape is fine for squat,
  clean, and deadlift.
- **Mark the bar with a centered dot or ring on the end cap.** Centered matters:
  sleeves spin during turnover, and off-center tape orbits the bar axis by ~2.5cm.
- **Take a test shot** and verify the full range of motion — floor to overhead
  lockout — is in frame before filming the set.
- For fault-detection validation, also film 2–3 **deliberate-fault reps** per
  movement (intentional early arm bend, cut depth, forward drift).

## Development

```sh
cd engine && uv run pytest      # engine suite (no model inference required)
cd engine && uv run ruff check  # lint
cd app && npx vitest run        # web app unit tests
cd app && npm run e2e           # Playwright end-to-end test
```

Layout:

```
engine/   Python analysis engine + FastAPI service (uv project)
  src/powerpath_engine/         geometry · series · decode · bar · calibration ·
                                pose · registry/ · segmentation · phases ·
                                metrics · faults · scoring · pipeline · overlay ·
                                cli · api/
  tools/                        click-to-label ground-truth tool + gate report
  tests/                        incl. synthetic lift generators + video renderers
app/      Next.js web UI (library · upload · analysis player)
docs/contracts/                 the frozen engine↔UI JSON contract
samples/                        generated demo clips (see make_samples.py)
run.sh                          one-command launcher
```

The engine↔UI interface is a frozen JSON contract
([docs/contracts/overlay-metrics-contract.md](docs/contracts/overlay-metrics-contract.md)),
enforced by shared test assertions on the Python side and typed clients on the
TypeScript side.

## Status & roadmap

**Built and working (M1–M2):** the full pipeline — decode → pose → bar tracking →
calibration → segmentation → phase detection → metrics, fault rules, made/missed,
0–100 rep scores → overlay JSON + annotated MP4 — running end-to-end on uploaded
video through the web app, with the 6 movements above.

**Roadmap:**

- [ ] **M3 — real-footage validation**: run the ground-truth filming set through
      the pose models, tune the two detectors that currently lean on synthetic
      trajectories, personal best-rep templates + "vs your best" + ghost-rep
      overlay.
- [ ] **M4 — full catalog**: the remaining 8 movements + thruster complex.
- [ ] **M5 — coaching**: Gemini-written coach notes (metrics JSON only — video
      never leaves the machine), progress charts, fault-frequency trends.
- [ ] **M6 — live mode**: camera at the rack, cues between reps, same engine.

## Privacy

All measurement happens on-device. The API binds `127.0.0.1` only. The one
planned cloud call (M5 coach notes) sends extracted metrics JSON — never video.
