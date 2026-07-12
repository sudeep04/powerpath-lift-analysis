/**
 * Typed client for the PowerPath engine API (FastAPI, 127.0.0.1:8400).
 * The UI never touches SQLite or the library directory directly — HTTP only.
 */

export type JobState = "QUEUED" | "RUNNING" | "DONE" | "FAILED";

export type JobStage = "decode" | "pose" | "bar" | "segment" | "metrics" | null;

export interface Job {
  state: JobState;
  progress: number;
  stage: JobStage;
  error: string | null;
}

export interface Movement {
  key: string;
  display_name: string;
  family: string;
}

export interface VideoSummary {
  video_id: string;
  movement: string;
  display_name: string;
  load_kg: number;
  filmed_at: string;
  job: Job;
  rep_count: number | null;
  best_score: number | null;
}

export interface UploadResponse {
  video_id: string;
  job_id: string;
}

export interface UploadParams {
  file: File;
  movement: string;
  loadKg: number;
  recalibrate: boolean;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function apiBase(): string {
  return process.env.NEXT_PUBLIC_API ?? "http://127.0.0.1:8400";
}

async function errorFrom(res: Response): Promise<ApiError> {
  let message = `Request failed (${res.status})`;
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object") {
      const detail = (body as Record<string, unknown>).detail;
      const error = (body as Record<string, unknown>).error;
      if (typeof detail === "string") message = detail;
      else if (typeof error === "string") message = error;
    }
  } catch {
    // non-JSON body; keep the default message
  }
  return new ApiError(message, res.status);
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, init);
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as T;
}

/** POST /api/videos — multipart upload; kicks off an analysis job. */
export async function uploadVideo(params: UploadParams): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", params.file);
  form.append("movement", params.movement);
  form.append("load_kg", String(params.loadKg));
  form.append("recalibrate", String(params.recalibrate));
  return requestJson<UploadResponse>("/api/videos", {
    method: "POST",
    body: form,
  });
}

/** GET /api/jobs/{id} — poll job state/progress/stage. */
export function getJob(jobId: string): Promise<Job> {
  return requestJson<Job>(`/api/jobs/${encodeURIComponent(jobId)}`);
}

/** GET /api/videos — library listing with embedded job snapshots. */
export function listVideos(): Promise<VideoSummary[]> {
  return requestJson<VideoSummary[]>("/api/videos");
}

/** GET /api/movements — registry entries for the movement select. */
export function listMovements(): Promise<Movement[]> {
  return requestJson<Movement[]>("/api/movements");
}

/** GET /api/videos/{id}/analysis — metrics JSON (consumed by Task 12). */
export function getAnalysis(videoId: string): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(
    `/api/videos/${encodeURIComponent(videoId)}/analysis`,
  );
}

/** GET /api/videos/{id}/overlay — overlay JSON (consumed by Task 12). */
export function getOverlay(videoId: string): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(
    `/api/videos/${encodeURIComponent(videoId)}/overlay`,
  );
}

/** Stable URL for the original video bytes (used as a <video> src). */
export function videoFileUrl(videoId: string): string {
  return `${apiBase()}/api/videos/${encodeURIComponent(videoId)}/file`;
}

/** GET /api/videos/{id}/file — original bytes (used by FAILED-card retry). */
export async function getVideoFile(videoId: string): Promise<Blob> {
  const res = await fetch(videoFileUrl(videoId));
  if (!res.ok) throw await errorFrom(res);
  return res.blob();
}

/** DELETE /api/videos/{id} */
export async function deleteVideo(videoId: string): Promise<void> {
  const res = await fetch(`${apiBase()}/api/videos/${encodeURIComponent(videoId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw await errorFrom(res);
}
