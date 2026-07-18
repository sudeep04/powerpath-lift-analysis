# PowerPath JSON contracts (FROZEN 2026-07-18)

The interface between the engine (Task 8 `overlay.py` writes these) and the UI
(Task 12 renders `overlay.json`; Task 10 serves both verbatim from disk). Both tasks
build to THIS. All times are PTS seconds (float). All positions are in the upright image
pixel space of the analyzed video (UI scales to its canvas). Angles in degrees, y-up
already applied for biomechanics values but overlay `bar`/`skeleton` pixel coords are in
IMAGE space (y-down, as the video renders) so the UI can draw directly.

## overlay.json  (GET /api/videos/{id}/overlay)
```json
{
  "video": { "width": 1920, "height": 1080, "fps_avg": 59.94, "duration_s": 12.34 },
  "movement": "power_clean",
  "frames": [
    {
      "t": 1.234,
      "bar": [512.0, 300.5],                     // [x,y] image px, or null if bar not tracked this frame
      "skeleton": {                               // landmark name -> [x,y] image px; omit/null a landmark if not detected
        "nose": [500.0, 120.0], "left_shoulder": [...], "right_shoulder": [...],
        "left_elbow": [...], "right_elbow": [...], "left_wrist": [...], "right_wrist": [...],
        "left_hip": [...], "right_hip": [...], "left_knee": [...], "right_knee": [...],
        "left_ankle": [...], "right_ankle": [...]
      }
    }
    // ... one entry per ANALYZED frame, strictly increasing t (pose may be 30fps-effective; bar 60fps — include every frame that has either; a missing bar is null, a missing/empty skeleton is {} — not null)
  ],
  "reps": [
    {
      "rep_index": 0,
      "t_start": 0.10, "t_end": 2.60,
      "made": true,
      "score": 84,                                // null for missed reps
      "bar_path": [[512.0,300.5],[510.0,280.0], ...],  // this rep's bar polyline, image px, for the trail overlay
      "phases": { "knee_pass": 0.55, "second_pull": 0.80, "catch": 1.25 },  // phase name -> t (only detected phases)
      "faults": [
        { "code": "bar_drift", "message": "Bar drifts 4.2cm forward at the knee", "phase": "knee_pass", "value": 4.2, "threshold": 3.0, "severity": "fault" }
      ],
      // severity is "fault" | "informational" (Task 7-fix added it to FaultFinding;
      // catch_above_parallel is "informational"). The UI mutes informational faults.
      "unanalyzed_reason": null                   // string if the rep was flagged unanalyzed, else null
    }
  ]
}
```

## metrics.json  (GET /api/videos/{id}/analysis)
```json
{
  "video": { "width": 1920, "height": 1080, "fps_avg": 59.94, "duration_s": 12.34 },
  "movement": "power_clean",
  "load_kg": 82.5,
  "extraction_version": 1,
  "rules_version": 1,
  "calibration": { "source": "plate", "bar_scale_cm_per_px": 0.225, "warning": null },
  "reps": [
    {
      "rep_index": 0,
      "made": true,
      "score": 84,
      "excluded_from_templates": false,
      "metrics": {
        "bar_drift_cm": 4.2,
        "peak_concentric_velocity_ms": 1.61,
        "path_length_ratio": 1.08,
        "smoothness": 0.82,
        "hip_angle_at_phase": { "knee_pass": 128.0, "catch": 95.0 },
        "knee_angle_at_phase": { "...": 0.0 },
        "elbow_angle_at_phase": { "...": 0.0 }
      },
      "faults": [ { "code": "...", "message": "...", "phase": "...", "value": 0.0, "threshold": 0.0 } ],
      "phases": { "knee_pass": 0.55, "catch": 1.25 }
    }
  ]
}
```

## Rules
- `overlay.json` is what the player canvas reads: `frames[]` for per-frame bar+skeleton
  (UI binary-searches by `t` against `video.currentTime`), `reps[].bar_path` for the
  per-rep trail, `reps[].phases`/`faults` for annotations at their timestamps,
  `reps[].score`/`made`/`unanalyzed_reason` for the filmstrip cards.
- `metrics.json` is the full analysis record (superset of numbers); the "vs best"
  section (M3) will consume it later.
- Missed reps: `made:false`, `score:null`, still present with their window + faults.
- Never emit NaN/Infinity; use null. Keys are snake_case. Landmark names match
  series.LANDMARK_NAMES.
- Task 10's FAKE engine (`POWERPATH_FAKE_ENGINE=1`) must write an overlay.json + metrics.json
  in EXACTLY this shape (a small canned but schema-valid sample: >=1 rep, a few frames)
  so Task 12's Playwright E2E renders a real overlay. Task 8 owns making the fake writer
  conform (update Task 10's fake runner if needed — that's an allowed cross-task fix).
