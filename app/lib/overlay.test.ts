import { describe, expect, it } from "vitest";
import {
  computeMapping,
  findFrame,
  frameIndexAtOrBefore,
  imageToCanvas,
  repAtTime,
  type OverlayDoc,
  type OverlayFrame,
  type OverlayRep,
} from "@/lib/overlay";

function frame(t: number): OverlayFrame {
  return { t, bar: [t * 100, 500], skeleton: null };
}

function overlayWith(frames: OverlayFrame[]): OverlayDoc {
  return {
    video: { width: 1920, height: 1080, fps_avg: 60, duration_s: 10 },
    movement: "power_clean",
    frames,
    reps: [],
  };
}

describe("findFrame binary search", () => {
  const frames = [frame(0.5), frame(1.0), frame(1.5), frame(2.0)];
  const overlay = overlayWith(frames);

  it("returns the frame on an exact t hit", () => {
    expect(findFrame(overlay, 1.0)).toBe(frames[1]);
    expect(findFrame(overlay, 0.5)).toBe(frames[0]);
    expect(findFrame(overlay, 2.0)).toBe(frames[3]);
  });

  it("returns the frame just-before t when between frames", () => {
    expect(findFrame(overlay, 1.2)).toBe(frames[1]);
    expect(findFrame(overlay, 1.4999)).toBe(frames[1]);
    expect(findFrame(overlay, 0.7)).toBe(frames[0]);
  });

  it("returns null before the first frame (documented choice)", () => {
    expect(findFrame(overlay, 0.2)).toBeNull();
    expect(findFrame(overlay, 0.4999)).toBeNull();
  });

  it("returns the last frame after the last t", () => {
    expect(findFrame(overlay, 2.0001)).toBe(frames[3]);
    expect(findFrame(overlay, 99)).toBe(frames[3]);
  });

  it("returns null for an empty frames array", () => {
    expect(findFrame(overlayWith([]), 1.0)).toBeNull();
  });

  it("stays consistent with a linear scan across a dense timeline", () => {
    const dense = Array.from({ length: 1001 }, (_, i) => frame(i * 0.016));
    for (const t of [0, 0.0159, 0.016, 4.4444, 7.9999, 16.0, 17.5]) {
      let expected = -1;
      for (let i = 0; i < dense.length; i++) {
        if (dense[i].t <= t) expected = i;
      }
      expect(frameIndexAtOrBefore(dense, t)).toBe(expected);
    }
  });
});

describe("repAtTime", () => {
  const rep = (i: number, t0: number, t1: number): OverlayRep => ({
    rep_index: i,
    t_start: t0,
    t_end: t1,
    made: true,
    score: 80,
    bar_path: [],
    phases: {},
    faults: [],
    unanalyzed_reason: null,
  });
  const reps = [rep(0, 0.1, 1.0), rep(1, 1.5, 2.4)];

  it("finds the rep containing t (inclusive bounds)", () => {
    expect(repAtTime(reps, 0.5)?.rep_index).toBe(0);
    expect(repAtTime(reps, 0.1)?.rep_index).toBe(0);
    expect(repAtTime(reps, 1.0)?.rep_index).toBe(0);
    expect(repAtTime(reps, 2.0)?.rep_index).toBe(1);
  });

  it("returns null between and outside reps", () => {
    expect(repAtTime(reps, 1.2)).toBeNull();
    expect(repAtTime(reps, 0.0)).toBeNull();
    expect(repAtTime(reps, 3.0)).toBeNull();
  });
});

describe("computeMapping / imageToCanvas (aspect-preserving contain fit)", () => {
  it("scales exactly with no offsets when aspects match", () => {
    const m = computeMapping(
      { width: 1920, height: 1080 },
      { width: 960, height: 540 },
    );
    expect(m).toEqual({ scale: 0.5, offsetX: 0, offsetY: 0 });
  });

  it("letterboxes vertically when the rect is taller than the video", () => {
    const m = computeMapping(
      { width: 1920, height: 1080 },
      { width: 960, height: 600 },
    );
    expect(m.scale).toBeCloseTo(0.5);
    expect(m.offsetX).toBeCloseTo(0);
    expect(m.offsetY).toBeCloseTo(30); // (600 - 540) / 2
  });

  it("pillarboxes horizontally for portrait video in a landscape rect", () => {
    const m = computeMapping(
      { width: 1080, height: 1920 },
      { width: 960, height: 540 },
    );
    expect(m.scale).toBeCloseTo(540 / 1920);
    expect(m.offsetY).toBeCloseTo(0);
    expect(m.offsetX).toBeCloseTo((960 - 1080 * (540 / 1920)) / 2);
  });

  it("maps image points through scale + offset", () => {
    const m = { scale: 0.5, offsetX: 10, offsetY: 20 };
    expect(imageToCanvas([100, 200], m)).toEqual([60, 120]);
    expect(imageToCanvas([0, 0], m)).toEqual([10, 20]);
  });

  it("returns a zero mapping (never NaN) for degenerate sizes", () => {
    const m = computeMapping({ width: 0, height: 0 }, { width: 800, height: 600 });
    expect(m).toEqual({ scale: 0, offsetX: 0, offsetY: 0 });
    const m2 = computeMapping({ width: 1920, height: 1080 }, { width: 0, height: 0 });
    expect(m2.scale).toBe(0);
    expect(Number.isNaN(m2.offsetX)).toBe(false);
  });
});
