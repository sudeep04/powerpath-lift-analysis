/**
 * Schema-valid overlay.json + metrics.json sample in the FROZEN contract
 * shape (.superpowers/sdd/overlay-metrics-contract.md). Shared by the Vitest
 * component tests and the Playwright E2E so both render the same document:
 * 5 reps — three made, one missed, one unanalyzed — with frames that include
 * null bar / null skeleton entries to exercise graceful degradation.
 */

import type {
  AnalysisDoc,
  OverlayDoc,
  OverlayFrame,
  Point,
  Skeleton,
} from "./overlay";

function skeletonAt(x: number, y: number): Skeleton {
  const p = (dx: number, dy: number): Point => [x + dx, y + dy];
  return {
    nose: p(0, -260),
    left_shoulder: p(-40, -200),
    right_shoulder: p(40, -200),
    left_elbow: p(-60, -140),
    right_elbow: p(60, -140),
    left_wrist: p(-70, -80),
    right_wrist: p(70, -80),
    left_hip: p(-30, -60),
    right_hip: p(30, -60),
    left_knee: p(-35, 60),
    right_knee: p(35, 60),
    left_ankle: p(-40, 180),
    right_ankle: p(40, 180),
  };
}

function buildFrames(): OverlayFrame[] {
  const frames: OverlayFrame[] = [];
  // 0.0s .. 6.0s at 10 analyzed frames/s — plenty for binary-search realism.
  for (let i = 0; i <= 60; i++) {
    const t = Math.round(i * 100) / 1000; // 0.0, 0.1, ... 6.0
    const barY = 700 - 400 * Math.abs(Math.sin((i / 60) * Math.PI * 5));
    const bar: Point | null = i % 13 === 7 ? null : [520 + (i % 5) * 2, barY];
    const skeleton = i % 9 === 4 ? null : skeletonAt(600, 640);
    frames.push({ t, bar, skeleton });
  }
  return frames;
}

function barPathFor(frames: OverlayFrame[], t0: number, t1: number): Point[] {
  return frames
    .filter((f) => f.t >= t0 && f.t <= t1 && f.bar !== null)
    .map((f) => f.bar as Point);
}

const FRAMES = buildFrames();

export const OVERLAY_FIXTURE: OverlayDoc = {
  video: { width: 1920, height: 1080, fps_avg: 59.94, duration_s: 6.5 },
  movement: "power_clean",
  frames: FRAMES,
  reps: [
    {
      rep_index: 0,
      t_start: 0.1,
      t_end: 1.0,
      made: true,
      score: 84,
      bar_path: barPathFor(FRAMES, 0.1, 1.0),
      phases: { knee_pass: 0.4, second_pull: 0.6, catch: 0.8 },
      faults: [
        {
          code: "bar_drift",
          message: "Bar drifts 4.2cm forward at the knee",
          phase: "knee_pass",
          value: 4.2,
          threshold: 3.0,
        },
      ],
      unanalyzed_reason: null,
    },
    {
      rep_index: 1,
      t_start: 1.2,
      t_end: 2.1,
      made: true,
      score: 91,
      bar_path: barPathFor(FRAMES, 1.2, 2.1),
      phases: { knee_pass: 1.5, second_pull: 1.7, catch: 1.9 },
      faults: [],
      unanalyzed_reason: null,
    },
    {
      rep_index: 2,
      t_start: 2.3,
      t_end: 3.2,
      made: false,
      score: null,
      bar_path: barPathFor(FRAMES, 2.3, 3.2),
      phases: { knee_pass: 2.6, second_pull: 2.8 },
      faults: [
        {
          code: "early_arm_bend",
          message: "Arms bend before full extension",
          phase: "second_pull",
          value: 24.0,
          threshold: 10.0,
        },
      ],
      unanalyzed_reason: null,
    },
    {
      rep_index: 3,
      t_start: 3.4,
      t_end: 4.3,
      made: true,
      score: 76,
      bar_path: barPathFor(FRAMES, 3.4, 4.3),
      phases: { knee_pass: 3.7, catch: 4.1 },
      faults: [
        {
          code: "slow_turnover",
          message: "Turnover slightly slower than your average",
          phase: "catch",
          value: 0.42,
          threshold: 0.35,
          severity: "informational",
        },
      ],
      unanalyzed_reason: null,
    },
    {
      rep_index: 4,
      t_start: 4.5,
      t_end: 5.6,
      made: false,
      score: null,
      bar_path: barPathFor(FRAMES, 4.5, 5.0),
      phases: {},
      faults: [],
      unanalyzed_reason: "bar marker lost during catch",
    },
  ],
};

export const ANALYSIS_FIXTURE: AnalysisDoc = {
  video: OVERLAY_FIXTURE.video,
  movement: "power_clean",
  load_kg: 82.5,
  extraction_version: 1,
  rules_version: 1,
  calibration: { source: "plate", bar_scale_cm_per_px: 0.225, warning: null },
  reps: OVERLAY_FIXTURE.reps.map((rep) => ({
    rep_index: rep.rep_index,
    made: rep.made,
    score: rep.score,
    excluded_from_templates: rep.unanalyzed_reason !== null,
    metrics:
      rep.unanalyzed_reason !== null
        ? {}
        : {
            bar_drift_cm: 4.2 - rep.rep_index * 0.3,
            peak_concentric_velocity_ms: 1.61 + rep.rep_index * 0.02,
            path_length_ratio: 1.08,
            smoothness: 0.82,
            hip_angle_at_phase: { knee_pass: 128.0, catch: 95.0 },
            knee_angle_at_phase: { knee_pass: 142.0 },
            elbow_angle_at_phase: { catch: 68.0 },
          },
    faults: rep.faults,
    phases: rep.phases,
  })),
};
