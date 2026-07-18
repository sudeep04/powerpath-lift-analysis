"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { videoFileUrl } from "@/lib/api";
import {
  computeMapping,
  fetchAnalysis,
  fetchOverlay,
  repAtTime,
  type AnalysisDoc,
  type CanvasMapping,
  type OverlayDoc,
  type Size,
} from "@/lib/overlay";
import {
  DEFAULT_THEME,
  drawOverlay,
  readOverlayTheme,
  type OverlayTheme,
} from "@/lib/overlay-renderer";
import { MetricsPanel } from "@/components/metrics-panel";
import { RepFilmstrip } from "@/components/rep-filmstrip";

function prettyMovement(key: string): string {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatLoad(loadKg: number): string {
  return `${Number.isInteger(loadKg) ? loadKg : loadKg.toFixed(1)} KG`;
}

/**
 * Analysis player: video + overlay canvas (left), per-rep metrics panel
 * (right), rep filmstrip (bottom). The canvas is redrawn from a
 * requestAnimationFrame loop synced to video.currentTime — findFrame binary
 * search picks the overlay frame, never frame-index math. With
 * prefers-reduced-motion the loop is replaced by static draws on seek/time
 * events.
 */
export function Player({ videoId }: { videoId: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const [overlay, setOverlay] = useState<OverlayDoc | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisDoc | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);

  // Refs mirror state/geometry for the imperative draw loop.
  const overlayRef = useRef<OverlayDoc | null>(null);
  const selectedRef = useRef(0);
  const mappingRef = useRef<CanvasMapping | null>(null);
  const sizeRef = useRef<Size | null>(null);
  const dprRef = useRef(1);
  const themeRef = useRef<OverlayTheme>(DEFAULT_THEME);

  useEffect(() => {
    overlayRef.current = overlay;
  }, [overlay]);
  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);

  // Load overlay.json (required) + metrics.json (optional, panel degrades).
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [overlayDoc, analysisDoc] = await Promise.all([
          fetchOverlay(videoId),
          fetchAnalysis(videoId).catch(() => null),
        ]);
        if (cancelled) return;
        setOverlay(overlayDoc);
        setAnalysis(analysisDoc);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load overlay");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [videoId]);

  // One overlay pass at the video's current time; also keeps the selected
  // rep in sync with playback (only fires a state update on rep change).
  const drawNow = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const overlayDoc = overlayRef.current;
    const mapping = mappingRef.current;
    const size = sizeRef.current;
    if (!video || !canvas || !overlayDoc || !mapping || !size) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return; // jsdom / lost context: skip drawing gracefully

    const dpr = dprRef.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const t = video.currentTime;
    drawOverlay(ctx, overlayDoc, t, mapping, size, themeRef.current);

    const rep = repAtTime(overlayDoc.reps, t);
    if (rep) {
      const index = overlayDoc.reps.indexOf(rep);
      if (index !== selectedRef.current) setSelected(index);
    }
  }, []);

  // Size the canvas backing store to the video's displayed rect (and DPR),
  // recomputing the image-px -> canvas mapping on resize.
  useEffect(() => {
    if (!overlay) return;
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;

    themeRef.current = readOverlayTheme(document);

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      dprRef.current = dpr;
      sizeRef.current = { width: rect.width, height: rect.height };
      mappingRef.current = computeMapping(overlay.video, {
        width: rect.width,
        height: rect.height,
      });
      drawNow();
    };

    resize();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(resize);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [overlay, drawNow]);

  // Render loop. Reduced motion: static draws on media events instead.
  useEffect(() => {
    if (!overlay) return;
    const video = videoRef.current;
    if (!video) return;

    const reduceMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reduceMotion || typeof requestAnimationFrame !== "function") {
      const events = ["loadedmetadata", "timeupdate", "seeked"] as const;
      for (const name of events) video.addEventListener(name, drawNow);
      drawNow();
      return () => {
        for (const name of events) video.removeEventListener(name, drawNow);
      };
    }

    let raf = 0;
    const loop = () => {
      drawNow();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [overlay, drawNow]);

  const seekTo = useCallback(
    (t: number) => {
      const video = videoRef.current;
      if (video) video.currentTime = t;
      drawNow();
    },
    [drawNow],
  );

  const selectRep = useCallback(
    (index: number) => {
      const overlayDoc = overlayRef.current;
      if (!overlayDoc) return;
      const rep = overlayDoc.reps[index];
      if (!rep) return;
      setSelected(index);
      selectedRef.current = index;
      seekTo(rep.t_start);
    },
    [seekTo],
  );

  if (error) {
    return (
      <div className="pp-cut border border-fail/60 bg-surface p-6">
        <h2 className="font-display text-lg font-bold uppercase tracking-[0.08em] text-fail">
          Analysis unavailable
        </h2>
        <p className="mt-2 font-mono text-sm text-muted">{error}</p>
      </div>
    );
  }

  if (!overlay) {
    return (
      <p
        role="status"
        className="font-mono text-sm uppercase tracking-[0.1em] text-muted"
      >
        Loading analysis…
      </p>
    );
  }

  const selectedRep = overlay.reps[selected] ?? null;
  const selectedAnalysisRep =
    (selectedRep &&
      analysis?.reps.find((r) => r.rep_index === selectedRep.rep_index)) ||
    null;

  return (
    <section className="pp-rise">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="font-display text-2xl font-bold uppercase tracking-[0.06em]">
          {prettyMovement(analysis?.movement ?? overlay.movement)}
        </h2>
        <p className="font-mono text-xs uppercase tracking-[0.12em] text-muted">
          {analysis != null && <>{formatLoad(analysis.load_kg)} · </>}
          {overlay.video.width}&times;{overlay.video.height} ·{" "}
          {overlay.video.duration_s.toFixed(1)}s
        </p>
      </header>

      {analysis?.calibration?.warning != null && (
        <p
          role="alert"
          className="pp-cut mt-4 border border-warn/60 bg-surface px-4 py-3 font-mono text-xs leading-relaxed text-warn"
        >
          <span className="uppercase tracking-[0.14em]">Calibration</span>{" "}
          &middot; {analysis.calibration.warning}
        </p>
      )}

      <div className="mt-6 grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        {/* Video + overlay canvas */}
        <div
          ref={wrapRef}
          className="pp-cut relative w-full self-start overflow-hidden border border-line bg-surface"
          style={{
            aspectRatio: `${overlay.video.width} / ${overlay.video.height}`,
          }}
        >
          <video
            ref={videoRef}
            src={videoFileUrl(videoId)}
            controls
            playsInline
            preload="metadata"
            className="absolute inset-0 h-full w-full"
          />
          <canvas
            ref={canvasRef}
            data-testid="overlay-canvas"
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
        </div>

        {/* Metrics for the selected rep */}
        {selectedRep ? (
          <MetricsPanel
            rep={selectedRep}
            analysisRep={selectedAnalysisRep}
            onSeek={seekTo}
          />
        ) : (
          <div className="pp-cut pp-blueprint border border-line bg-surface p-6">
            <p className="font-mono text-xs text-muted">
              No reps detected in this video.
            </p>
          </div>
        )}
      </div>

      {overlay.reps.length > 0 && (
        <RepFilmstrip
          reps={overlay.reps}
          selected={selected}
          onSelect={selectRep}
        />
      )}
    </section>
  );
}
