/**
 * Canvas drawing for the analysis player overlay. Pure functions of
 * (ctx, overlay, t, mapping) so they can be unit-tested with a mock 2D
 * context. Every channel degrades gracefully: null bar, null/partial
 * skeleton, missing phases, and empty bar paths simply draw nothing.
 *
 * Colors come from the Chalk & Iron tokens (globals.css); `readOverlayTheme`
 * resolves the CSS variables at runtime with the same token values as
 * fallbacks — no new colors are introduced here.
 */

import {
  findFrame,
  frameIndexAtOrBefore,
  imageToCanvas,
  repAtTime,
  type CanvasMapping,
  type Fault,
  type OverlayDoc,
  type OverlayRep,
  type Point,
  type Size,
  type Skeleton,
} from "@/lib/overlay";

export interface OverlayTheme {
  accent: string;
  ink: string;
  muted: string;
  warn: string;
  fail: string;
  bg: string;
}

/** Chalk & Iron token values, used as fallbacks for the CSS variables. */
export const DEFAULT_THEME: OverlayTheme = {
  accent: "#c8f135",
  ink: "#e8ecf1",
  muted: "#8b94a3",
  warn: "#ffb020",
  fail: "#ff5c5c",
  bg: "#0b0d10",
};

/** Resolve the design tokens from :root, falling back to the same values. */
export function readOverlayTheme(doc: Document): OverlayTheme {
  const styles = doc.defaultView?.getComputedStyle(doc.documentElement);
  const read = (name: string, fallback: string): string => {
    const value = styles?.getPropertyValue(name).trim();
    return value || fallback;
  };
  return {
    accent: read("--pp-accent", DEFAULT_THEME.accent),
    ink: read("--pp-ink", DEFAULT_THEME.ink),
    muted: read("--pp-muted", DEFAULT_THEME.muted),
    warn: read("--pp-warn", DEFAULT_THEME.warn),
    fail: read("--pp-fail", DEFAULT_THEME.fail),
    bg: read("--pp-bg", DEFAULT_THEME.bg),
  };
}

/** Skeleton bone chains: shoulder->hip->knee->ankle both sides, plus arms. */
export const SKELETON_BONES: ReadonlyArray<readonly [string, string]> = [
  ["left_shoulder", "right_shoulder"],
  ["left_hip", "right_hip"],
  ["left_shoulder", "left_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_shoulder", "right_hip"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
];

/** How long a fault callout stays on screen after its phase timestamp. */
export const FAULT_WINDOW_BEFORE_S = 0.05;
export const FAULT_WINDOW_AFTER_S = 1.2;

/** Frames of lookback for the bar marker when the current frame's bar is
 *  null (mirrors the engine's <=5-frame interpolation rule, with slack). */
const BAR_LOOKBACK_FRAMES = 8;

const TRAIL_MIN_ALPHA = 0.12;
const TRAIL_MAX_ALPHA = 0.85;

export interface VisibleFault {
  fault: Fault;
  /** 1 at the phase timestamp, fading to 0 at the end of the window. */
  alpha: number;
}

/**
 * Faults whose phase timestamp is near `t`. A fault is visible from just
 * before its phase time until FAULT_WINDOW_AFTER_S after it, fading out.
 * Faults whose phase is missing from rep.phases are skipped.
 */
export function faultsVisibleAt(rep: OverlayRep, t: number): VisibleFault[] {
  const visible: VisibleFault[] = [];
  for (const fault of rep.faults) {
    const phaseT = rep.phases[fault.phase];
    if (phaseT === undefined) continue;
    if (t < phaseT - FAULT_WINDOW_BEFORE_S) continue;
    if (t > phaseT + FAULT_WINDOW_AFTER_S) continue;
    const age = Math.max(0, t - phaseT);
    visible.push({ fault, alpha: 1 - (age / FAULT_WINDOW_AFTER_S) * 0.7 });
  }
  return visible;
}

/**
 * Current bar anchor in image px: the frame's bar, or the most recent
 * non-null bar within BAR_LOOKBACK_FRAMES, else null.
 */
export function barAnchorAt(overlay: OverlayDoc, t: number): Point | null {
  const idx = frameIndexAtOrBefore(overlay.frames, t);
  if (idx < 0) return null;
  const stop = Math.max(0, idx - BAR_LOOKBACK_FRAMES);
  for (let i = idx; i >= stop; i--) {
    const bar = overlay.frames[i].bar;
    if (bar) return bar;
  }
  return null;
}

/**
 * Dashed, fading accent trail of the current rep's bar path, drawn up to the
 * rep-progress point at `t`. bar_path carries no timestamps, so progress is
 * approximated linearly across [t_start, t_end]; older segments fade out.
 */
export function drawBarTrail(
  ctx: CanvasRenderingContext2D,
  rep: OverlayRep,
  t: number,
  mapping: CanvasMapping,
  theme: OverlayTheme,
): void {
  const path = rep.bar_path;
  if (path.length < 2) return;
  const span = rep.t_end - rep.t_start;
  const fraction =
    span > 0 ? Math.min(1, Math.max(0, (t - rep.t_start) / span)) : 1;
  const end = Math.round(fraction * (path.length - 1));
  if (end < 1) return;

  ctx.save();
  ctx.strokeStyle = theme.accent;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash([6, 4]);
  for (let i = 1; i <= end; i++) {
    const [x0, y0] = imageToCanvas(path[i - 1], mapping);
    const [x1, y1] = imageToCanvas(path[i], mapping);
    ctx.globalAlpha =
      TRAIL_MIN_ALPHA + (TRAIL_MAX_ALPHA - TRAIL_MIN_ALPHA) * (i / end);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
  ctx.restore();
}

/** Filled accent dot + soft ring at the current bar position. */
export function drawBarMarker(
  ctx: CanvasRenderingContext2D,
  bar: Point,
  mapping: CanvasMapping,
  theme: OverlayTheme,
): void {
  const [x, y] = imageToCanvas(bar, mapping);
  ctx.save();
  ctx.fillStyle = theme.accent;
  ctx.globalAlpha = 1;
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = theme.accent;
  ctx.globalAlpha = 0.35;
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

/** Hairline ink skeleton: bone chains plus small joint dots. */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  skeleton: Skeleton,
  mapping: CanvasMapping,
  theme: OverlayTheme,
): void {
  ctx.save();
  ctx.strokeStyle = theme.ink;
  ctx.lineWidth = 1.5;
  ctx.lineCap = "round";
  ctx.setLineDash([]);
  ctx.globalAlpha = 0.55;
  for (const [a, b] of SKELETON_BONES) {
    const pa = skeleton[a];
    const pb = skeleton[b];
    if (!pa || !pb) continue;
    const [x0, y0] = imageToCanvas(pa, mapping);
    const [x1, y1] = imageToCanvas(pb, mapping);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
  ctx.fillStyle = theme.ink;
  ctx.globalAlpha = 0.7;
  for (const point of Object.values(skeleton)) {
    if (!point) continue;
    const [x, y] = imageToCanvas(point, mapping);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/**
 * Callout text near the bar for faults active at `t`. Informational faults
 * render muted; real faults render warn. Text is kept inside the canvas
 * horizontally and stacked when several faults are active at once.
 */
export function drawFaultCallouts(
  ctx: CanvasRenderingContext2D,
  rep: OverlayRep,
  t: number,
  anchor: Point,
  size: Size,
  theme: OverlayTheme,
): void {
  const visible = faultsVisibleAt(rep, t);
  if (visible.length === 0) return;

  ctx.save();
  ctx.font = '500 11px "IBM Plex Mono", ui-monospace, monospace';
  ctx.textBaseline = "middle";
  const padX = 6;
  const boxH = 20;
  let stack = 0;
  for (const { fault, alpha } of visible) {
    const tone =
      fault.severity === "informational" ? theme.muted : theme.warn;
    const text = fault.message;
    const textW = ctx.measureText(text).width;
    const boxW = textW + padX * 2;
    let x = anchor[0] + 16;
    if (x + boxW > size.width - 4) x = Math.max(4, anchor[0] - 16 - boxW);
    const y = anchor[1] - 18 - stack * (boxH + 4);

    ctx.globalAlpha = 0.78 * alpha;
    ctx.fillStyle = theme.bg;
    ctx.fillRect(x, y - boxH / 2, boxW, boxH);
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = tone;
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.strokeRect(x, y - boxH / 2, boxW, boxH);
    ctx.fillStyle = tone;
    ctx.fillText(text, x + padX, y);
    stack += 1;
  }
  ctx.restore();
}

/**
 * One full overlay pass for `t` (video.currentTime): clear, then rep bar
 * trail, skeleton, bar marker, and fault callouts. Any missing channel is
 * skipped. `size` is the canvas rect in CSS px (the ctx transform handles
 * devicePixelRatio).
 */
export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  overlay: OverlayDoc,
  t: number,
  mapping: CanvasMapping,
  size: Size,
  theme: OverlayTheme = DEFAULT_THEME,
): void {
  ctx.clearRect(0, 0, size.width, size.height);
  if (mapping.scale <= 0) return;

  const frame = findFrame(overlay, t);
  const rep = repAtTime(overlay.reps, t);

  if (rep) drawBarTrail(ctx, rep, t, mapping, theme);
  if (frame?.skeleton) drawSkeleton(ctx, frame.skeleton, mapping, theme);

  const bar = barAnchorAt(overlay, t);
  if (bar) drawBarMarker(ctx, bar, mapping, theme);

  if (rep) {
    const anchor =
      bar ?? (rep.bar_path.length > 0 ? rep.bar_path[rep.bar_path.length - 1] : null);
    if (anchor) {
      drawFaultCallouts(ctx, rep, t, imageToCanvas(anchor, mapping), size, theme);
    }
  }
}
