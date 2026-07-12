/**
 * The list endpoint embeds job snapshots but not job ids; only POST /api/videos
 * returns a job_id. We remember video_id -> job_id in sessionStorage so the
 * library can poll GET /api/jobs/{id} at 1s while a job is QUEUED/RUNNING.
 * When a mapping is missing (fresh tab), the library falls back to re-fetching
 * the list, which carries refreshed job snapshots.
 */

const KEY = "powerpath.active-jobs";

function read(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.sessionStorage.getItem(KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, string>;
    }
  } catch {
    // corrupted storage — start clean
  }
  return {};
}

function write(map: Record<string, string>): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(KEY, JSON.stringify(map));
  } catch {
    // storage unavailable (private mode quota etc.) — polling falls back
  }
}

export function rememberJob(videoId: string, jobId: string): void {
  const map = read();
  map[videoId] = jobId;
  write(map);
}

export function forgetJob(videoId: string): void {
  const map = read();
  if (videoId in map) {
    delete map[videoId];
    write(map);
  }
}

export function getActiveJobs(): Record<string, string> {
  return read();
}
