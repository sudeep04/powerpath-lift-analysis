# PowerPath M1+M2 Implementation Plan

Source of truth for product decisions: the approved design doc at
`~/.gstack/projects/2026POC/sudeepreddy-unknown-design-20260711-141133.md`.
This plan decomposes milestones M1 (engine CLI) and M2 (minimal app) into tasks.

## Global Constraints (binding for every task)

- **Streaming single-pass rule:** the pipeline never buffers decoded frames. PyAV
  yields frames one at a time; pose, bar tracking, and overlay data collection are
  per-frame consumers; only extracted time series (a few MB) survive the pass. Peak
  RSS under ~500MB on a 30s 60fps 1080p video.
- **PTS timebase:** every engine output (per-frame landmarks, bar positions, rep
  boundaries, phase landmarks) is keyed by PTS seconds (float, from PyAV), never by
  frame index alone. iPhone video is variable frame rate.
- **`engine/src/powerpath_engine/geometry.py` is the SOLE owner of coordinate
  conversions:** px↔cm for bar plane and body plane, image-y flip (image y grows
  down; all biomechanics values use y-up), joint-angle helpers. No other module
  touches raw scale factors.
- **Two-plane calibration:** bar-plane scale from a 450mm plate (or manual/date
  fallback); body-plane scale from athlete height. Bar-vs-body relationships are
  expressed as angles/ratios, never raw cross-plane cm.
- **Marker-only bar tracking:** color-blob detection of a centered dot/ring on the
  bar end cap. Gaps ≤5 frames: interpolate; longer gaps in a rep's critical phases:
  rep marked `unanalyzed` with a reason string. No fallback tracker.
- **No model inference in the test suite:** pose runs behind a `PoseBackend`
  interface; tests use `FakePoseBackend` and synthetic trajectory generators.
  Golden fixtures come from a fixture-extraction script run manually later.
- **Versions:** analyses record `extraction_version` and `rules_version`
  (module-level constants; bump rules documented in code comments).
- **Jobs:** `ProcessPoolExecutor(max_workers=1)` for analysis; SQLite `jobs` table
  with states QUEUED → RUNNING (progress 0-100, stage) → DONE | FAILED (error).
  No Celery/Redis. FastAPI binds **127.0.0.1 only**.
- **SQLite is owned by the FastAPI layer.** The UI reads via HTTP API only.
- **Storage:** `~/PowerPath/library/{YYYY-MM-DD}/{video_id}/` for originals +
  `metrics.json` + `overlay.json`. Overridable via `POWERPATH_LIBRARY` env var
  (tests use tmp dirs).
- **Stacks:** Python 3.12 managed by `uv` (pyproject in `engine/`), ruff + pytest.
  Next.js (App Router, TypeScript) in `app/`, Vitest for unit tests, Playwright
  (chromium only) for E2E. Engine deps: numpy, opencv-python, av (PyAV), scipy,
  fastapi, uvicorn, python-multipart, hypothesis + pytest (dev). rtmlib and
  mediapipe are OPTIONAL extras imported lazily — the suite must pass without them.
- **Commit style:** conventional commits (`feat:`, `test:`, `chore:`); commit per
  logical unit; never `git add -A` from repo root (engine and app have separate
  ignores).
- **Movement scope for M1:** 5 family representatives — power_clean, back_squat,
  push_press, power_snatch, deadlift. The registry design must make the other 9
  v1 movements config-only additions.

## Repository Layout (target)

```
powerpath/
├── docs/plans/m1-m2-implementation.md   (this file)
├── engine/                    # Python: analysis engine + API
│   ├── pyproject.toml
│   ├── src/powerpath_engine/
│   │   ├── geometry.py        # Task 1
│   │   ├── series.py          # Task 2 (time series + smoothing)
│   │   ├── decode.py          # Task 3
│   │   ├── bar.py             # Task 4 (marker tracking)
│   │   ├── calibration.py     # Task 4
│   │   ├── pose.py            # Task 5 (PoseBackend + impls)
│   │   ├── registry/          # Task 6 (movement configs)
│   │   ├── segmentation.py    # Task 6
│   │   ├── phases.py          # Task 6
│   │   ├── metrics.py         # Task 7
│   │   ├── faults.py          # Task 7
│   │   ├── scoring.py         # Task 7
│   │   ├── pipeline.py        # Task 8
│   │   ├── overlay.py         # Task 8 (overlay JSON + annotated MP4)
│   │   ├── cli.py             # Task 8
│   │   ├── versions.py        # Task 8 (extraction_version, rules_version)
│   │   └── api/               # Task 10 (FastAPI, db, jobs, storage)
│   ├── tools/label.py         # Task 9
│   ├── tools/gate_report.py   # Task 9
│   └── tests/
├── app/                       # Next.js UI (Tasks 11-12)
└── run.sh                     # Task 13
```

## Task 1: Engine scaffold + geometry module

Create `engine/` with uv-managed pyproject (name `powerpath-engine`, package
`powerpath_engine` under `src/`), ruff config, pytest + hypothesis dev deps, and
repo-root `.gitignore` covering Python (.venv, __pycache__, .pytest_cache, dist),
Node (node_modules, .next), `.superpowers/`, `.DS_Store`.

Implement `geometry.py`:
- `class PlaneScale(cm_per_px: float)` with `px_to_cm(px)` and `cm_to_px(cm)`.
- `bar_plane_scale_from_plate(plate_diameter_px: float) -> PlaneScale` using the
  450mm standard plate diameter (45.0 cm).
- `body_plane_scale_from_height(athlete_height_cm: float, athlete_height_px: float)
  -> PlaneScale`.
- `to_y_up(y_px: float, frame_height_px: int) -> float` and inverse — all
  biomechanics math uses y-up; image space is y-down.
- `joint_angle(a, b, c) -> float` — interior angle at b in degrees for 2D points,
  stable for collinear/degenerate input (return 180.0 for collinear, raise
  ValueError only on coincident points).
- `horizontal_deviation_cm(x_px_series: list[float], scale: PlaneScale) ->
  list[float]` — signed deviation from the first sample.

Tests (pytest + hypothesis):
- Property: px→cm→px round-trip identity (both planes, random scales 0.01-1.0).
- Property: to_y_up is an involution given frame height.
- joint_angle: right angle = 90°, straight = 180°, known 45° case; degenerate cases.
- bar_plane_scale_from_plate: 450px plate → 1.0 mm/px sanity (45cm/450px).

## Task 2: Time series core + smoothing

`series.py`:
- `@dataclass Sample(t: float, x: float, y: float, visibility: float = 1.0)` —
  t is PTS seconds; x/y in px (image space) at this layer.
- `@dataclass TimeSeries(samples: list[Sample])` with: `ts()`, `xs()`, `ys()`
  (numpy arrays); `interpolate_gaps(max_gap_frames: int, expected_dt: float) ->
  tuple[TimeSeries, list[Gap]]` where `Gap(t_start, t_end, filled: bool)` — linear
  interpolation for gaps ≤ max_gap_frames·expected_dt, longer gaps left unfilled
  and reported; `smooth(window_s: float = 0.15) -> TimeSeries` — Savitzky-Golay
  (polyorder 2, window derived from median dt, forced odd, ≥5); `velocity() ->
  numpy array` (central differences on smoothed y, units are caller's, per
  second); `slice_time(t0, t1)`.
- `@dataclass LandmarkFrame(t: float, points: dict[str, Sample])` and
  `LandmarkSeries` with the same gap/smooth API applied per landmark name.
  Landmark names (subset used downstream): nose, left/right shoulder, elbow,
  wrist, hip, knee, ankle, heel, foot_index.
- All operations pure (return new objects).

Tests: gap interpolation fills a 3-frame hole exactly linearly and reports a
6-frame hole unfilled; SG smoothing reduces added white noise variance on a sine
≥5x while preserving peak location within 1 sample; velocity of a linear ramp is
constant within tolerance; slice_time boundaries inclusive-exclusive.

## Task 3: Streaming decode

`decode.py`:
- `class VideoMeta(width, height, rotation_deg, fps_avg, duration_s)`.
- `probe(path) -> VideoMeta` — reads container/stream metadata via PyAV; rotation
  from stream side data / display matrix (0/90/180/270).
- `frames(path) -> Iterator[DecodedFrame]` where `DecodedFrame(t: float, image:
  np.ndarray BGR, index: int)` — a GENERATOR that decodes lazily (never
  materializes the video), applies rotation normalization so the returned image
  is always upright, converts to BGR uint8, and sets t from `frame.pts *
  stream.time_base` (float seconds). Raises `DecodeError` (custom) on unreadable
  files or streams with no video.
- `write_test_video(path, frames: list[np.ndarray], fps: int)` helper in
  `tests/video_utils.py` (PyAV h264 encode) used across the suite.

Tests: encode a 2s synthetic video (moving white square), assert monotonically
increasing t, frame count within 10% of fps·duration, upright dimensions after
encoding a rotated stream (simulate by encoding portrait dimensions), DecodeError
on a text file and on a truncated file (write half the bytes of a valid mp4).
Memory: iterating a 10s 720p test video keeps RSS growth under 150MB (use
resource.getrusage before/after; generous bound, CI-safe).

## Task 4: Bar marker tracking + plate calibration

`bar.py`:
- `MarkerSpec(hsv_low: tuple, hsv_high: tuple)` with a default for a saturated
  pink/magenta marker (document: user tapes a centered dot/ring on the end cap).
- `detect_marker(image_bgr, spec, roi: Rect | None) -> Sample | None` — HSV
  threshold → morphological open → largest blob → centroid + `visibility` =
  blob_area / expected_area clamped to [0,1]; None when no blob ≥ 20px area.
- `track(frames_iter) -> TimeSeries` is NOT here — bar.py stays per-frame
  (pipeline owns iteration, per the streaming rule). Provide
  `class MarkerTracker` with `feed(t, image) -> Sample | None` that also
  maintains a search ROI around the last hit (3× blob bbox) for speed, resetting
  to full frame after 5 misses.
- `estimate_marker_diameter_px(samples) -> float` for the scale sanity
  cross-check (bar sleeve ≈ 50mm).

`calibration.py`:
- `detect_plate_circle(image_bgr) -> tuple[center, radius_px] | None` — Hough
  circles on a downscaled gray frame; largest circle whose radius is 15-45% of
  frame height.
- `class CalibrationResult(bar_scale: PlaneScale, source: Literal["plate",
  "date_fallback", "manual"], warning: str | None)`.
- `calibrate(first_frames: list[np.ndarray], date_fallback: PlaneScale | None,
  manual: PlaneScale | None, marker_diameter_px: float | None) ->
  CalibrationResult` — plate scale must sit inside the plausible band
  (0.5-3.0 mm/px) AND agree within 20% with the marker-derived 50mm sleeve check
  when marker_diameter_px given; otherwise fall back (date → manual → error) with
  a human-readable warning. Never silently produce a wrong scale.

Tests (synthetic images drawn with OpenCV): marker centroid within 1px of a drawn
dot under gaussian noise + motion blur (cv2.blur 9x1); ROI tracker follows a dot
moving 15px/frame and recovers after 5 blank frames; plate detection finds a drawn
450px circle within 2% radius; calibrate rejects a scale from a 100px "background
plate" (out of band) and falls back with warning; sanity cross-check rejects
plate scale disagreeing >20% with marker diameter.

## Task 5: Pose backend interface

`pose.py`:
- `class PoseBackend(Protocol)`: `detect(image_bgr) -> dict[str, Sample] | None`
  (landmark name → Sample with x/y px + visibility 0-1; t filled by caller).
- `class FakePoseBackend(script: Callable[[int], dict[str, Sample] | None])` —
  deterministic, call-counted; used by all tests.
- `class RTMLibBackend` — lazy `import rtmlib` inside `__init__`; uses
  `rtmlib.Body` (balanced mode, onnxruntime backend); maps COCO-17 keypoints to
  our landmark names; picks the detection nearest the previous athlete bbox
  (single-athlete lock — a background person must not steal the track); raises
  `PoseUnavailableError` with an install hint if rtmlib is missing.
- `class MediaPipeBackend` — same contract, lazy import, `pose_landmarker` task
  API; optional.
- `class StridedPose(backend, stride: int = 2)` — `feed(t, image, index)` runs the
  backend every `stride` frames, records misses as None; and
  `rerun_full_rate(window_frames: Iterable[DecodedFrame]) -> LandmarkSeries` for
  rep-window full-rate re-runs (pipeline supplies a second decode pass restricted
  to windows — decode is cheap relative to pose; document this).
- ROI: `crop_around(prev_bbox, image, pad=0.3)` helper; backends receive crops and
  returned Samples are mapped back to full-frame coordinates by StridedPose.

Tests use FakePoseBackend only: stride 2 calls backend exactly ceil(n/2) times;
ROI mapping returns full-frame coords (feed a fake that asserts crop size);
single-athlete lock chooses nearest-to-previous bbox given two scripted people;
PoseUnavailableError message names the extra (`uv add rtmlib onnxruntime`).

## Task 6: Movement registry + segmentation + phases

`registry/` — one module per movement + `base.py`:
- `@dataclass MovementConfig(key, display_name, family: Literal["squat","clean",
  "press","snatch","hinge"], starts_from: Literal["floor","hang","rack",
  "shoulders"], bar_travel: Literal["up","down_up","up_down"], phases:
  list[PhaseDef], fault_rules: list[str], made_criteria: MadeCriteria,
  comparison_landmarks: list[str])`.
- `PhaseDef(name, detector: str, params: dict)` — detector is a named strategy
  resolved in `phases.py` (explicit over clever; no lambdas in config).
- Registry configs for: `power_clean`, `back_squat`, `push_press`,
  `power_snatch`, `deadlift` (per design-doc phase sketches: clean = setup/
  first_pull/knee_pass/second_pull/catch/recovery; snatch = same pulls, receive =
  bar peak ABOVE head landmark with elbow lockout; squat = descent/bottom/ascent/
  standing; push press = dip/drive/press_out/lockout/standing; deadlift = setup/
  knee_pass/lockout/descent, no catch). `registry.get(key)` +
  `registry.all_keys()` — adding a movement is one new config module.

`segmentation.py`:
- `segment(bar: TimeSeries, config: MovementConfig, expected_dt: float) ->
  list[RepWindow]` where `RepWindow(t_start, t_end, rep_index)`.
- Movement-aware state machine: IDLE → ACTIVE on sustained |vy| above
  `v_start=0.15 m/s` (in calibrated m/s — segmentation receives a bar series
  already converted to cm by the pipeline) for ≥120ms AND total displacement ≥
  `min_disp` (config: 20cm squat/deadlift, 40cm clean/snatch/press); ACTIVE →
  IDLE when |vy| < 0.05 m/s for ≥400ms at a y within 15cm of the rep's start
  band (squat/deadlift/press return to start; clean/snatch end at rack/overhead —
  use config.bar_travel to pick the end condition). Catch-dip zero crossings and
  intra-rep pauses must NOT split a rep (hysteresis + the 400ms dwell). A
  terminal fast free-fall (vy < -2.5 m/s sustained) is the bar drop → discard,
  never a rep.
- `synthetic.py` (in tests/): trajectory generators producing bar TimeSeries +
  matching LandmarkSeries for: clean N reps with catch dip, squat N reps with
  walkout jitter (±2cm noise before first rep), dumped clean (free-fall from
  catch height), zero-rep video (jitter only), push press with dip, deadlift
  up-down, snatch with overhead receive. Parameterized (n_reps, load-ish speed,
  noise). These are the fixture source for Tasks 6-8 tests.

`phases.py`: named detectors — `bar_leaves_floor`, `knee_pass` (bar y crosses
knee landmark y), `hip_contact` (bar nearest hip landmark), `peak_hip_extension_
velocity` (hip angle angular velocity peak, needs LandmarkSeries),
`catch_rack` (bar vy zero after peak + elbow angle rotates ≥40° within 200ms),
`receive_overhead` (bar y above nose landmark + elbow angle ≥165°), `bottom`
(bar y minimum in window), `lockout_top` (bar y max with hip+knee ≥170°),
`dip_turnaround` (local bar minimum ≤15cm deep). Each returns t (PTS) or None;
`detect_phases(rep, bar, landmarks, config) -> dict[str, float | None]`.

Tests (all on synthetic generators): clean 5 reps → exactly 5 windows despite
catch dips; squat with walkout jitter → correct count; dumped clean → rep
excluded by free-fall rule; zero-rep → []; single-rep video → 1; hang variant
config (starts_from="hang") does not require bar_leaves_floor (add a minimal
`hang_power_clean` config to prove registry extensibility); phase detectors on
clean synthetic: knee_pass/hip_contact/catch within ±2 frames of generator
ground truth; snatch receive_overhead fires only when bar above head landmark;
deadlift produces no catch phase and lockout_top present.

## Task 7: Metrics, fault rules, quality score

`metrics.py` — per RepWindow, given calibrated bar series (cm), landmarks,
phases: `bar_drift_cm` (max |x deviation| from setup vertical, per phase),
`peak_concentric_velocity_ms`, `path_length_ratio` (path length / net vertical),
`smoothness_normalized_jerk` (dimensionless normalized jerk of bar trajectory),
angles at each phase landmark (hip, knee, elbow via geometry.joint_angle),
`catch_height_ratio` (bar y at catch / athlete height proxy = hip-to-ankle px).

`faults.py` — rule = pure function `(RepMetrics, MovementConfig) ->
FaultFinding | None` with `FaultFinding(code, message, phase, value, threshold)`.
Rules for the 5 representatives (thresholds in config params, not code):
early_arm_bend (elbow < 170° before peak_hip_extension t), bar_drift
(drift > envelope cm from config: clean/snatch 6cm, squat 4cm, deadlift 5cm,
press 4cm), squat_depth (hip landmark y-up above knee y-up at bottom → shallow),
early_press_out (press family: elbow extension begins before dip_turnaround +
drive complete), catch_above_parallel (power clean/snatch: hip angle at receive
< 90° → it became a squat rep — informational), no_lockout (press/snatch/deadlift:
hip/knee/elbow angles at lockout below 170°).

`scoring.py` — made/missed from config.made_criteria (clean: catch_rack phase
found AND recovery to standing; press/snatch: receive/lockout found with elbows
≥165°; squat/deadlift: returns to start band without free-fall). Score (made reps
only): smoothness 30 + path efficiency 30 + velocity 20 + faults 20 (each fault
−7, floor 0). Velocity component compares peak_concentric_velocity to the
athlete's history at ±10% load via a `VelocityHistory` protocol (API supplies
it; tests fake it); with <5 same-load reps redistribute the 20 to smoothness
(+10) and path (+10) — document formula in module docstring. Missed reps: no
score, `excluded_from_templates=True`.

Tests: every fault rule has a triggering and non-triggering synthetic fixture;
made/missed on dumped clean = missed; score redistribution kicks in below 5
history reps; score is 0-100 clamped; deterministic given fixed inputs.

## Task 8: Pipeline orchestrator + overlay + CLI

`versions.py`: `EXTRACTION_VERSION = 1`, `RULES_VERSION = 1` + docstring on bump
rules (extraction: decode/pose/bar/calibration change; rules: segmentation/
faults/scoring/templates change).

`pipeline.py`:
- `analyze(path, movement_key, load_kg, athlete_height_cm, pose_backend,
  date_fallback_scale=None, manual_scale=None, progress_cb=None) ->
  AnalysisResult`.
- Single streaming pass: decode.frames → MarkerTracker.feed + StridedPose.feed
  per frame (calibration frames grabbed from the first 30), never storing
  frames. Then: calibrate, convert bar series px→cm (geometry), smooth,
  segment, second restricted decode pass for rep-window full-rate pose,
  phases, metrics, faults, scoring. `progress_cb(stage: str, pct: int)` at
  stage boundaries (stages: decode, pose, bar, segment, metrics).
- `AnalysisResult(video: VideoMeta, calibration: CalibrationResult, reps:
  list[RepResult], extraction_version, rules_version, unanalyzed:
  list[UnanalyzedRep])` where `RepResult(window, made, score, metrics, faults,
  phases)`, serialized by `overlay.py`.

`overlay.py`:
- `write_metrics_json(result, path)` — full analysis (schema documented in
  docstring, stable key names).
- `write_overlay_json(result, bar_series, landmark_series, path)` — PTS-keyed
  per-frame data for the UI: `{"frames": [{"t": 1.234, "bar": [x,y], "skeleton":
  {name: [x,y], ...}}], "reps": [...], "bar_path_by_rep": {...}}` (px in upright
  image space; UI scales to its canvas).
- `write_annotated_mp4(video_path, overlay_data, out_path)` — second decode
  pass drawing bar trail + skeleton + rep/fault text via OpenCV; streaming
  (VideoWriter frame by frame).

`cli.py` (argparse, `powerpath` console script): `analyze VIDEO --movement KEY
--load KG [--height CM] [--pose fake|rtmlib|mediapipe] [--out DIR]` → writes
metrics.json, overlay.json, annotated.mp4, prints rep table (rep, made, score,
top fault). `extract-fixtures VIDEO --movement KEY --out DIR` → freezes bar +
landmark series JSON as golden fixtures. `movements` → lists registry keys.

Integration tests: synthetic 5-rep clean video (generated frames with a drawn
marker dot following the synthetic trajectory; FakePoseBackend scripted from the
matching landmark series) run end-to-end through `analyze` → 5 reps, all made,
metrics.json + overlay.json schema-valid (validate required keys), overlay t
values strictly increasing, annotated.mp4 exists and has ≥0.9× input frame
count; RSS growth during analyze < 500MB (resource.getrusage); progress_cb
called with all five stages in order.

## Task 9: Click-to-label tool + gate report

`tools/label.py` (runs with engine venv, cv2 GUI): `python tools/label.py VIDEO
--keyframes setup,knee,hip,receive,standing` — steps to each requested position
via trackbar/arrow keys, user clicks the bar-end center, writes
`VIDEO.labels.json`: `{"video": ..., "clicks": [{"name": "knee", "t": 1.23,
"frame_index": 74, "x": 512, "y": 301}]}`. Esc aborts; u undoes last click.
~150 lines, no engine imports beyond decode.

`tools/gate_report.py`: `python tools/gate_report.py VIDEO.labels.json
overlay.json --scale-mm-per-px F` → prints per-keyframe |bar - label| in px and
cm, rep-count comparison, PASS/FAIL against the M1 gates (≤1cm at keyframes),
exit code 0/1. Unit test with hand-built JSONs (no GUI test for label.py; keep
its logic-free of cv2 where testable: JSON writing behind a function).

## Task 10: FastAPI service + SQLite + job worker

`api/` package: `db.py` (sqlite3, WAL mode, schema DDL exactly as the design
doc's data model sketch — videos, calibrations, jobs, analyses, reps, templates,
settings; plus `schema_version` pragma), `storage.py` (library layout
`~/PowerPath/library/{date}/{video_id}/original.<ext>`, `POWERPATH_LIBRARY` env
override), `jobs.py` (ProcessPoolExecutor(1); submit runs
`pipeline.analyze` in the worker process writing progress rows back via its own
sqlite connection; on API startup, RUNNING jobs from a dead process →
re-queued), `main.py` (FastAPI app factory `create_app(engine_runner=...)` —
runner injectable so API tests use a fake that writes canned results).

Endpoints (JSON): `POST /api/videos` (multipart: file, movement, load_kg,
recalibrate: bool) → validates movement against registry, stores file, creates
job → `{video_id, job_id}`; `GET /api/jobs/{id}` → `{state, progress, stage,
error}`; `GET /api/videos` → list with latest job state + rep summary; `GET
/api/videos/{id}/analysis` → metrics.json content; `GET
/api/videos/{id}/overlay` → overlay.json content; `GET /api/videos/{id}/file` →
video file (range requests via FileResponse); `DELETE /api/videos/{id}`;
`GET /api/movements` → registry keys + display names. CORS allow
http://localhost:3000. Uvicorn entry `powerpath-api` binds 127.0.0.1:8400.

Tests (httpx TestClient + fake runner + tmp library): upload happy path →
QUEUED→DONE with fake results retrievable; invalid movement 422; corrupted
upload → job FAILED with error surfaced; double-submit same file → two distinct
jobs queued sequentially (pool size 1); restart re-queue (simulate by writing a
RUNNING row with stale pid, call create_app, assert QUEUED); DELETE removes rows
+ files; movements endpoint lists 5+ keys.

## Task 11: Next.js scaffold + upload + library

`app/` via create-next-app (TypeScript, App Router, no src dir is fine,
Tailwind yes). `.env.local` → `NEXT_PUBLIC_API=http://127.0.0.1:8400`. Shared
`lib/api.ts` typed client (fetch wrappers for the Task 10 endpoints; types
mirror API JSON).

Pages: `/` library — grid of videos (movement, date, load, job state chip,
rep-count + best score once DONE; FAILED shows error reason + Retry button
(re-POST); empty state: "Drop your first video" hero). `/upload` (or modal from
`/`) — drag-drop file input, movement dropdown fed from `GET /api/movements`
(all 14 will appear as registry grows; shows the 5 for now), load kg number
field, recalibrate toggle, submit → navigate to library; while a job is
QUEUED/RUNNING poll `GET /api/jobs/{id}` every 1s and render progress bar +
stage label.

Vitest + Testing Library: api client URL/method mapping; upload form validates
(no file / no movement / load ≤0 disable submit); library renders states
(empty, RUNNING with progress, FAILED with retry, DONE with score) from mocked
fetch. Layout/hierarchy per the approved wireframe (video-left, metrics-right
comes in Task 12; keep library visual style minimal dark athletic per design
doc mood).

## Task 12: Player + overlay canvas + filmstrip

`/video/[id]` page: `<video>` element (src = file endpoint) with a canvas
overlay absolutely positioned; `lib/overlay.ts` — `findFrame(overlay, t)`
binary search on PTS (≤16ms lookup), scale px→canvas mapping preserving aspect;
draws: bar path trail for current rep (dashed, fades older points), current bar
marker, skeleton lines (shoulder-hip-knee-ankle chains + arms), fault
annotations at their phase timestamps (callout text near bar). RequestAnimationFrame
loop synced to `video.currentTime`. Rep filmstrip bottom: one card per rep
(score, made/missed, unanalyzed reason when present); click seeks video to
rep.t_start; prev/next rep buttons. Metrics panel right: per-rep metrics +
faults + phases; "vs best" section placeholder (M3).

Vitest: findFrame binary search (exact hit, between frames, before first, after
last); px→canvas scaling math; filmstrip seek computes correct time. Playwright
(chromium): with API running in fake-engine mode (env `POWERPATH_FAKE_ENGINE=1`
supported by Task 10's app factory — canned instant results) — upload a tiny
generated mp4 → library shows DONE → player page renders canvas + filmstrip
with 5 reps. Keep the E2E to this single happy path.

## Task 13: run.sh + README + smoke wiring

`run.sh`: starts API (`uv run powerpath-api`) and app (`npm run dev`) with
cleanup trap, waits for both ports, opens http://localhost:3000. `README.md`:
quickstart (uv sync, npm install, ./run.sh), the filming assignment summary
(from design doc: 5 representatives, portrait for overhead, centered ring
marker, test shot, deliberate-fault reps), CLI usage, architecture sketch
(reuse design doc diagram), test commands. Verify: fresh-clone dry run
instructions actually work (script checks uv/node present with friendly
errors). Final `git log` sanity: conventional commits throughout.
