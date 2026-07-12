"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteVideo,
  getJob,
  getVideoFile,
  listVideos,
  uploadVideo,
  type VideoSummary,
} from "@/lib/api";
import { forgetJob, getActiveJobs, rememberJob } from "@/lib/active-jobs";
import { VideoCard } from "@/components/video-card";

const POLL_MS = 1000;

function isActive(video: VideoSummary): boolean {
  return video.job.state === "QUEUED" || video.job.state === "RUNNING";
}

export function Library() {
  const [videos, setVideos] = useState<VideoSummary[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retryingId, setRetryingId] = useState<string | null>(null);
  const [retryError, setRetryError] = useState<string | null>(null);

  const videosRef = useRef<VideoSummary[] | null>(null);
  useEffect(() => {
    videosRef.current = videos;
  }, [videos]);

  const refresh = useCallback(async () => {
    try {
      setVideos(await listVideos());
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Failed to load library");
    }
  }, []);

  // Initial load — deferred to a promise callback so state updates happen
  // outside the synchronous effect body (react-hooks/set-state-in-effect).
  useEffect(() => {
    void Promise.resolve().then(refresh);
  }, [refresh]);

  const hasActive = videos?.some(isActive) ?? false;

  // While any job is QUEUED/RUNNING, poll GET /api/jobs/{id} every 1s for the
  // jobs we know ids for; refresh the whole list when a job finishes or when
  // an active video has no known job id (fresh tab).
  useEffect(() => {
    if (!hasActive) return;
    let cancelled = false;

    const tick = async () => {
      const current = videosRef.current;
      if (!current) return;
      const jobIds = getActiveJobs();
      const active = current.filter(isActive);
      let needListRefresh = false;

      await Promise.all(
        active.map(async (video) => {
          const jobId = jobIds[video.video_id];
          if (!jobId) {
            needListRefresh = true;
            return;
          }
          try {
            const job = await getJob(jobId);
            if (cancelled) return;
            if (job.state === "DONE" || job.state === "FAILED") {
              forgetJob(video.video_id);
              needListRefresh = true;
            }
            setVideos((prev) =>
              prev
                ? prev.map((v) =>
                    v.video_id === video.video_id ? { ...v, job } : v,
                  )
                : prev,
            );
          } catch {
            needListRefresh = true;
          }
        }),
      );

      if (needListRefresh && !cancelled) await refresh();
    };

    const timer = setInterval(() => void tick(), POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [hasActive, refresh]);

  // Retry a FAILED video: re-POST the original bytes with the same movement
  // and load, then delete the failed record.
  const handleRetry = useCallback(
    async (video: VideoSummary) => {
      setRetryingId(video.video_id);
      setRetryError(null);
      try {
        const blob = await getVideoFile(video.video_id);
        const file = new File([blob], `${video.movement}-retry.mp4`, {
          type: blob.type || "video/mp4",
        });
        const res = await uploadVideo({
          file,
          movement: video.movement,
          loadKg: video.load_kg,
          recalibrate: false,
        });
        rememberJob(res.video_id, res.job_id);
        forgetJob(video.video_id);
        await deleteVideo(video.video_id);
        await refresh();
      } catch (err) {
        setRetryError(err instanceof Error ? err.message : "Retry failed");
      } finally {
        setRetryingId(null);
      }
    },
    [refresh],
  );

  if (loadError) {
    return (
      <div className="pp-cut border border-fail/60 bg-surface p-6">
        <h2 className="font-display text-lg font-bold uppercase tracking-[0.08em] text-fail">
          Engine unreachable
        </h2>
        <p className="mt-2 font-mono text-sm text-muted">{loadError}</p>
        <button
          type="button"
          onClick={() => void refresh()}
          className="pp-cut-sm mt-4 border border-line px-4 py-1.5 font-display text-sm font-semibold uppercase tracking-[0.14em] text-ink hover:border-accent/40"
        >
          Reload
        </button>
      </div>
    );
  }

  if (videos === null) {
    return (
      <p
        role="status"
        className="font-mono text-sm uppercase tracking-[0.1em] text-muted"
      >
        Loading library…
      </p>
    );
  }

  if (videos.length === 0) {
    return (
      <section className="pp-cut pp-blueprint flex min-h-[420px] flex-col items-center justify-center border border-line bg-surface px-6 py-16 text-center">
        <h2 className="font-display text-4xl font-bold uppercase tracking-[0.06em]">
          Drop your first video
        </h2>
        <p className="mt-3 max-w-md font-mono text-sm text-muted">
          Film side-on with the full bar path in frame — plate marker visible,
          one athlete under the bar.
        </p>
        <Link
          href="/upload"
          className="pp-cut mt-8 inline-block bg-accent px-6 py-2.5 font-display text-sm font-bold uppercase tracking-[0.14em] text-bg transition-[filter] hover:brightness-110"
        >
          Upload a lift
        </Link>
      </section>
    );
  }

  return (
    <section>
      <div className="mb-6 flex items-baseline justify-between">
        <h2 className="font-display text-2xl font-bold uppercase tracking-[0.06em]">
          Library
        </h2>
        <span className="font-mono text-xs uppercase tracking-[0.12em] text-muted">
          {videos.length} {videos.length === 1 ? "video" : "videos"}
        </span>
      </div>
      {retryError && (
        <p className="mb-4 font-mono text-xs text-fail">{retryError}</p>
      )}
      <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {videos.map((video, i) => (
          <li
            key={video.video_id}
            className="pp-rise"
            style={{ animationDelay: `${i * 40}ms` }}
          >
            <VideoCard
              video={video}
              onRetry={handleRetry}
              retrying={retryingId === video.video_id}
            />
          </li>
        ))}
      </ul>
    </section>
  );
}
