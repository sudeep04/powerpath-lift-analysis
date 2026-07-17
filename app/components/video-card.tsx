"use client";

import Link from "next/link";
import type { VideoSummary } from "@/lib/api";
import { StateChip } from "@/components/state-chip";

function formatDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  })
    .format(date)
    .toUpperCase();
}

function formatLoad(loadKg: number): string {
  return `${Number.isInteger(loadKg) ? loadKg : loadKg.toFixed(1)} KG`;
}

function formatScore(score: number): string {
  return Number.isInteger(score) ? String(score) : score.toFixed(1);
}

export function VideoCard({
  video,
  onRetry,
  retrying = false,
}: {
  video: VideoSummary;
  onRetry?: (video: VideoSummary) => void;
  retrying?: boolean;
}) {
  const { job } = video;
  const active = job.state === "QUEUED" || job.state === "RUNNING";

  return (
    <article className="pp-cut pp-card flex h-full flex-col border border-line bg-surface p-4 hover:border-accent/40">
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-display text-xl font-bold uppercase leading-tight tracking-[0.04em]">
          {video.display_name}
        </h3>
        <StateChip state={job.state} />
      </div>

      <div className="mt-2 flex items-baseline gap-3">
        <span className="font-mono text-sm text-ink">
          {formatLoad(video.load_kg)}
        </span>
        <span className="font-mono text-xs text-muted">
          {formatDate(video.filmed_at)}
        </span>
      </div>

      {active && (
        <div className="mt-auto pt-4">
          <div
            role="progressbar"
            aria-valuenow={job.progress}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Analysis progress"
            className="h-0.5 w-full bg-line"
          >
            <div
              className="pp-pulse h-full bg-accent"
              style={{ width: `${Math.min(100, Math.max(0, job.progress))}%` }}
            />
          </div>
          <p className="mt-2 font-mono text-xs uppercase tracking-[0.1em] text-muted">
            {job.stage ?? "queued"} · {job.progress}%
          </p>
        </div>
      )}

      {job.state === "DONE" && (
        <div className="mt-auto flex items-end gap-6 border-t border-line pt-3">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
              Reps
            </p>
            <p className="font-display text-2xl font-bold leading-none">
              {video.rep_count ?? "—"}
            </p>
          </div>
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
              Best score
            </p>
            <p className="font-display text-2xl font-bold leading-none text-accent">
              {video.best_score != null ? formatScore(video.best_score) : "—"}
            </p>
          </div>
          <Link
            href={`/video/${encodeURIComponent(video.video_id)}`}
            className="ml-auto font-display text-xs font-semibold uppercase tracking-[0.14em] text-ink transition-colors hover:text-accent"
          >
            Analysis &rsaquo;
          </Link>
        </div>
      )}

      {job.state === "FAILED" && (
        <div className="mt-auto border-t border-line pt-3">
          <p className="font-mono text-xs text-fail">
            {job.error ?? "Analysis failed"}
          </p>
          {onRetry && (
            <button
              type="button"
              disabled={retrying}
              onClick={() => onRetry(video)}
              className="pp-cut-sm mt-3 border border-fail/60 px-3 py-1 font-display text-xs font-semibold uppercase tracking-[0.14em] text-fail transition-colors hover:bg-fail/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {retrying ? "Retrying…" : "Retry"}
            </button>
          )}
        </div>
      )}
    </article>
  );
}
