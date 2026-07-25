# Sample videos for testing

These are **synthetic** clips (a magenta bar-end marker + a 450mm plate disc on a
plain background) rendered so the real engine can analyze them without real pose
models or real footage. Regenerate anytime with:

```bash
cd engine && uv run python tests/make_samples.py
```

## What's here

- `power_clean/power_clean.mp4` — input clip (480x640, 13.2s, 60fps, portrait)
- `power_clean/annotated.mp4` — the same clip with bar path + skeleton + rep/fault overlays drawn on
- `power_clean/metrics.json`, `overlay.json` — the full analysis output
- `back_squat/…` — same set for a squat (see caveat below)

**power_clean** is the full demo: 5 reps detected, all made, scores 84/79/79/79/79,
real phases (first_pull, knee_pass, second_pull, catch), calibration from the plate.

**back_squat** exercises decode → bar-tracking → calibration → segmentation (5 reps
found), but every rep reads as **missed / no score**. That is expected, not a bug:
the squat "made" rule requires the knee bent past 90° at the bottom, and the
synthetic body doesn't actually bend its knees (its joint angles barely move). On
**real** footage where the knee bends, squat scoring works. Use power_clean to see
the scored path; back_squat shows the tracking/segmentation path.

## Three ways to test

1. **Full app UI (recommended)** — canned-but-rich overlay, works with ANY uploaded video:
   ```bash
   POWERPATH_FAKE_ENGINE=1 ./run.sh
   ```
   Then open http://localhost:3000, upload `power_clean/power_clean.mp4` (or any mp4),
   watch the job run, and open the player: overlay canvas, rep filmstrip, metrics panel.
   (Fake-engine mode returns a canned analysis so you can exercise the whole UI without
   pose models.)

2. **Inspect a real engine analysis** — open the pre-generated files in `power_clean/`:
   play `annotated.mp4` to see the drawn bar path + skeleton, or read `metrics.json`.

3. **Run the real engine yourself (needs pose models)**:
   ```bash
   cd engine && uv sync --extra pose      # installs rtmlib + onnxruntime
   uv run powerpath analyze ../samples/power_clean/power_clean.mp4 \
     --movement power_clean --load 60 --out /tmp/pp_out
   ```
   Without pose models, `--pose fake` runs but produces no skeleton (bar-only analysis).
   The scored `samples/power_clean/*` were produced by the pipeline with pose scripted
   from the synthetic body (what `tests/make_samples.py` does).

> Real barbell footage is your M1 filming assignment — these synthetic clips are for
> exercising the app now, not for validating real-lift accuracy (that's the M3 gate).
