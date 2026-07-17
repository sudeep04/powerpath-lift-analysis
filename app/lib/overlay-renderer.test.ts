import { describe, expect, it, vi } from "vitest";
import type { OverlayDoc, OverlayRep } from "@/lib/overlay";
import {
  barAnchorAt,
  DEFAULT_THEME,
  drawOverlay,
  FAULT_WINDOW_AFTER_S,
  faultsVisibleAt,
} from "@/lib/overlay-renderer";

type MockCtx = CanvasRenderingContext2D & {
  arc: ReturnType<typeof vi.fn>;
  moveTo: ReturnType<typeof vi.fn>;
  lineTo: ReturnType<typeof vi.fn>;
  setLineDash: ReturnType<typeof vi.fn>;
  clearRect: ReturnType<typeof vi.fn>;
  fillText: ReturnType<typeof vi.fn>;
  fillRect: ReturnType<typeof vi.fn>;
};

function mockCtx(): MockCtx {
  return {
    save: vi.fn(),
    restore: vi.fn(),
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    arc: vi.fn(),
    setLineDash: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    fillText: vi.fn(),
    measureText: vi.fn(() => ({ width: 80 })),
    setTransform: vi.fn(),
    strokeStyle: "",
    fillStyle: "",
    globalAlpha: 1,
    lineWidth: 1,
    lineCap: "butt",
    lineJoin: "miter",
    font: "",
    textBaseline: "alphabetic",
  } as unknown as MockCtx;
}

const REP: OverlayRep = {
  rep_index: 0,
  t_start: 1.0,
  t_end: 2.0,
  made: true,
  score: 84,
  bar_path: [
    [100, 800],
    [110, 600],
    [120, 400],
    [130, 300],
  ],
  phases: { knee_pass: 1.3, catch: 1.8 },
  faults: [
    {
      code: "bar_drift",
      message: "Bar drifts forward",
      phase: "knee_pass",
      value: 4.2,
      threshold: 3.0,
    },
  ],
  unanalyzed_reason: null,
};

function docWith(overrides?: Partial<OverlayDoc>): OverlayDoc {
  return {
    video: { width: 1920, height: 1080, fps_avg: 60, duration_s: 5 },
    movement: "power_clean",
    frames: [
      { t: 1.0, bar: [100, 800], skeleton: null },
      {
        t: 1.5,
        bar: [120, 400],
        skeleton: {
          left_shoulder: [200, 300],
          left_hip: [210, 500],
          left_knee: [220, 700],
          left_ankle: [230, 900],
        },
      },
      { t: 2.0, bar: null, skeleton: null },
    ],
    reps: [REP],
    ...overrides,
  };
}

const SIZE = { width: 960, height: 540 };
// 1920x1080 -> 960x540: scale 0.5, no offsets.
const MAPPING = { scale: 0.5, offsetX: 0, offsetY: 0 };

describe("faultsVisibleAt", () => {
  it("shows a fault from its phase timestamp and fades it over the window", () => {
    const atPhase = faultsVisibleAt(REP, 1.3);
    expect(atPhase).toHaveLength(1);
    expect(atPhase[0].fault.code).toBe("bar_drift");
    expect(atPhase[0].alpha).toBeCloseTo(1);

    const later = faultsVisibleAt(REP, 1.3 + FAULT_WINDOW_AFTER_S / 2);
    expect(later).toHaveLength(1);
    expect(later[0].alpha).toBeLessThan(1);
    expect(later[0].alpha).toBeGreaterThan(0);
  });

  it("hides faults outside their window", () => {
    expect(faultsVisibleAt(REP, 1.0)).toHaveLength(0);
    expect(faultsVisibleAt(REP, 1.3 + FAULT_WINDOW_AFTER_S + 0.01)).toHaveLength(0);
  });

  it("skips faults whose phase is not in rep.phases", () => {
    const rep: OverlayRep = {
      ...REP,
      faults: [{ ...REP.faults[0], phase: "second_pull" }],
    };
    expect(faultsVisibleAt(rep, 1.3)).toHaveLength(0);
  });
});

describe("barAnchorAt", () => {
  it("uses the current frame's bar when present", () => {
    expect(barAnchorAt(docWith(), 1.5)).toEqual([120, 400]);
  });

  it("falls back to the last non-null bar within the lookback", () => {
    expect(barAnchorAt(docWith(), 2.0)).toEqual([120, 400]);
  });

  it("returns null before the first frame", () => {
    expect(barAnchorAt(docWith(), 0.1)).toBeNull();
  });
});

describe("drawOverlay", () => {
  it("clears the canvas and draws marker, skeleton, and dashed trail at scaled coords", () => {
    const ctx = mockCtx();
    drawOverlay(ctx, docWith(), 1.5, MAPPING, SIZE, DEFAULT_THEME);

    expect(ctx.clearRect).toHaveBeenCalledWith(0, 0, 960, 540);
    // Bar marker at image [120,400] -> canvas [60,200].
    expect(ctx.arc).toHaveBeenCalledWith(60, 200, 5, 0, Math.PI * 2);
    // Skeleton bone left_shoulder [200,300] -> left_hip [210,500], scaled.
    expect(ctx.moveTo).toHaveBeenCalledWith(100, 150);
    expect(ctx.lineTo).toHaveBeenCalledWith(105, 250);
    // Trail is dashed.
    expect(ctx.setLineDash).toHaveBeenCalledWith([6, 4]);
    // Trail starts at the first bar_path point [100,800] -> [50,400].
    expect(ctx.moveTo).toHaveBeenCalledWith(50, 400);
  });

  it("draws the fault callout near the bar at its phase timestamp", () => {
    const ctx = mockCtx();
    drawOverlay(ctx, docWith(), 1.3, MAPPING, SIZE, DEFAULT_THEME);
    expect(ctx.fillText).toHaveBeenCalledWith(
      "Bar drifts forward",
      expect.any(Number),
      expect.any(Number),
    );
  });

  it("degrades gracefully when bar and skeleton are null", () => {
    const doc = docWith({
      frames: [{ t: 1.5, bar: null, skeleton: null }],
    });
    const ctx = mockCtx();
    expect(() =>
      drawOverlay(ctx, doc, 1.5, MAPPING, SIZE, DEFAULT_THEME),
    ).not.toThrow();
    // No marker and no joints -> arc never called; trail still drawn.
    expect(ctx.arc).not.toHaveBeenCalled();
    expect(ctx.setLineDash).toHaveBeenCalledWith([6, 4]);
  });

  it("draws nothing but the clear for an empty document", () => {
    const doc = docWith({ frames: [], reps: [] });
    const ctx = mockCtx();
    drawOverlay(ctx, doc, 1.0, MAPPING, SIZE, DEFAULT_THEME);
    expect(ctx.clearRect).toHaveBeenCalledTimes(1);
    expect(ctx.arc).not.toHaveBeenCalled();
    expect(ctx.moveTo).not.toHaveBeenCalled();
    expect(ctx.fillText).not.toHaveBeenCalled();
  });

  it("skips drawing entirely on a degenerate (zero) mapping", () => {
    const ctx = mockCtx();
    drawOverlay(
      ctx,
      docWith(),
      1.5,
      { scale: 0, offsetX: 0, offsetY: 0 },
      SIZE,
      DEFAULT_THEME,
    );
    expect(ctx.clearRect).toHaveBeenCalledTimes(1);
    expect(ctx.arc).not.toHaveBeenCalled();
  });
});
