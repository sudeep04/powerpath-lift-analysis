/**
 * Types + lookup/scaling helpers for overlay.json and metrics.json
 * (frozen contract: .superpowers/sdd/overlay-metrics-contract.md).
 *
 * All times are PTS seconds (float). `bar`/`skeleton`/`bar_path` coordinates
 * are IMAGE pixels (y-down, as the video renders) — the UI scales them to the
 * displayed canvas rect with `computeMapping`/`imageToCanvas`.
 */

import { getAnalysis, getOverlay } from "@/lib/api";

export type Point = [number, number];

export interface OverlayVideoInfo {
  width: number;
  height: number;
  fps_avg: number;
  duration_s: number;
}

/** landmark name -> [x, y] image px; a landmark may be omitted or null. */
export type Skeleton = Partial<Record<string, Point | null>>;

export interface OverlayFrame {
  t: number;
  bar: Point | null;
  skeleton: Skeleton | null;
}

export interface Fault {
  code: string;
  message: string;
  phase: string;
  value: number | null;
  threshold: number | null;
  /**
   * Not part of the frozen v1 shape, but tolerated if the engine adds it:
   * "informational" faults are styled muted instead of warn/fail.
   */
  severity?: string | null;
}

export interface OverlayRep {
  rep_index: number;
  t_start: number;
  t_end: number;
  made: boolean;
  score: number | null;
  bar_path: Point[];
  /** phase name -> t (only detected phases). */
  phases: Record<string, number>;
  faults: Fault[];
  unanalyzed_reason: string | null;
}

export interface OverlayDoc {
  video: OverlayVideoInfo;
  movement: string;
  frames: OverlayFrame[];
  reps: OverlayRep[];
}

/* ---------------------------- metrics.json ---------------------------- */

export interface AnalysisRepMetrics {
  bar_drift_cm?: number | null;
  peak_concentric_velocity_ms?: number | null;
  path_length_ratio?: number | null;
  smoothness?: number | null;
  hip_angle_at_phase?: Record<string, number>;
  knee_angle_at_phase?: Record<string, number>;
  elbow_angle_at_phase?: Record<string, number>;
}

export interface AnalysisRep {
  rep_index: number;
  made: boolean;
  score: number | null;
  excluded_from_templates?: boolean;
  metrics: AnalysisRepMetrics;
  faults: Fault[];
  phases: Record<string, number>;
}

export interface AnalysisDoc {
  video: OverlayVideoInfo;
  movement: string;
  load_kg: number;
  extraction_version: number;
  rules_version: number;
  calibration: {
    source: string;
    bar_scale_cm_per_px: number | null;
    warning: string | null;
  };
  reps: AnalysisRep[];
}

/* ------------------------------ fetchers ------------------------------ */

/** GET /api/videos/{id}/overlay, typed against the frozen contract. */
export function fetchOverlay(videoId: string): Promise<OverlayDoc> {
  return getOverlay(videoId) as unknown as Promise<OverlayDoc>;
}

/** GET /api/videos/{id}/analysis, typed against the frozen contract. */
export function fetchAnalysis(videoId: string): Promise<AnalysisDoc> {
  return getAnalysis(videoId) as unknown as Promise<AnalysisDoc>;
}

/* ---------------------------- frame lookup ---------------------------- */

/**
 * Greatest index i with frames[i].t <= t, or -1 when t precedes every frame.
 * Plain binary search over the strictly-increasing `t` column — O(log n),
 * comfortably under the 16ms/frame budget for any realistic clip.
 */
export function frameIndexAtOrBefore(frames: OverlayFrame[], t: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

/**
 * Frame at/just-before `t` (video.currentTime).
 *
 * Documented edge choices:
 * - `t` before the first frame -> null (nothing is known yet; the renderer
 *   draws nothing rather than showing a future pose).
 * - `t` at/after the last frame -> the last frame.
 * - empty `frames` -> null.
 */
export function findFrame(overlay: OverlayDoc, t: number): OverlayFrame | null {
  const idx = frameIndexAtOrBefore(overlay.frames, t);
  return idx >= 0 ? overlay.frames[idx] : null;
}

/**
 * The rep whose [t_start, t_end] window contains `t` (inclusive on both
 * ends), or null between reps. Reps are few (<~30), a linear scan is fine.
 */
export function repAtTime(reps: OverlayRep[], t: number): OverlayRep | null {
  for (const rep of reps) {
    if (t >= rep.t_start && t <= rep.t_end) return rep;
  }
  return null;
}

/* --------------------------- px -> canvas map -------------------------- */

export interface Size {
  width: number;
  height: number;
}

export interface CanvasMapping {
  scale: number;
  offsetX: number;
  offsetY: number;
}

/**
 * Aspect-preserving "contain" fit of the native video frame (image px) into
 * the displayed canvas rect (CSS px) — the same math `object-fit: contain`
 * uses on the <video>, so overlay geometry lands on the pixels it annotates.
 * Degenerate dimensions produce a zero mapping (never NaN).
 */
export function computeMapping(video: Size, rect: Size): CanvasMapping {
  if (video.width <= 0 || video.height <= 0 || rect.width <= 0 || rect.height <= 0) {
    return { scale: 0, offsetX: 0, offsetY: 0 };
  }
  const scale = Math.min(rect.width / video.width, rect.height / video.height);
  return {
    scale,
    offsetX: (rect.width - video.width * scale) / 2,
    offsetY: (rect.height - video.height * scale) / 2,
  };
}

/** Map a point from image px to displayed canvas CSS px. */
export function imageToCanvas(point: Point, mapping: CanvasMapping): Point {
  return [
    point[0] * mapping.scale + mapping.offsetX,
    point[1] * mapping.scale + mapping.offsetY,
  ];
}
