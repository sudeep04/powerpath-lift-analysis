import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  deleteVideo,
  getAnalysis,
  getJob,
  getOverlay,
  getVideoFile,
  listMovements,
  listVideos,
  uploadVideo,
  videoFileUrl,
} from "@/lib/api";

const BASE = "http://127.0.0.1:8400";

type Call = { url: string; method: string; body: unknown };

let calls: Call[];
let responder: (url: string, method: string) => unknown;

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    json: async () => data,
    blob: async () => new Blob(["bytes"], { type: "video/quicktime" }),
  } as unknown as Response;
}

beforeEach(() => {
  calls = [];
  responder = () => ({});
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      calls.push({ url, method, body: init?.body });
      const data = responder(url, method);
      if (data instanceof Error) throw data;
      if (
        data &&
        typeof data === "object" &&
        "__status" in (data as Record<string, unknown>)
      ) {
        const { __status, ...rest } = data as Record<string, unknown>;
        return jsonResponse(rest, __status as number);
      }
      return jsonResponse(data);
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client URL/method mapping", () => {
  it("listVideos GETs /api/videos", async () => {
    responder = () => [];
    await expect(listVideos()).resolves.toEqual([]);
    expect(calls).toEqual([
      { url: `${BASE}/api/videos`, method: "GET", body: undefined },
    ]);
  });

  it("getJob GETs /api/jobs/{id}", async () => {
    const job = { state: "RUNNING", progress: 40, stage: "pose", error: null };
    responder = () => job;
    await expect(getJob("job-123")).resolves.toEqual(job);
    expect(calls[0]).toMatchObject({
      url: `${BASE}/api/jobs/job-123`,
      method: "GET",
    });
  });

  it("listMovements GETs /api/movements", async () => {
    responder = () => [
      { key: "power_clean", display_name: "Power Clean", family: "clean" },
    ];
    const movements = await listMovements();
    expect(movements[0].key).toBe("power_clean");
    expect(calls[0]).toMatchObject({
      url: `${BASE}/api/movements`,
      method: "GET",
    });
  });

  it("getAnalysis and getOverlay GET the per-video endpoints", async () => {
    responder = () => ({ ok: true });
    await getAnalysis("vid-1");
    await getOverlay("vid-1");
    expect(calls.map((c) => c.url)).toEqual([
      `${BASE}/api/videos/vid-1/analysis`,
      `${BASE}/api/videos/vid-1/overlay`,
    ]);
    expect(calls.every((c) => c.method === "GET")).toBe(true);
  });

  it("videoFileUrl builds the file URL; getVideoFile fetches bytes", async () => {
    expect(videoFileUrl("vid-1")).toBe(`${BASE}/api/videos/vid-1/file`);
    const blob = await getVideoFile("vid-1");
    expect(blob.type).toBe("video/quicktime");
    expect(calls[0]).toMatchObject({
      url: `${BASE}/api/videos/vid-1/file`,
      method: "GET",
    });
  });

  it("deleteVideo DELETEs /api/videos/{id}", async () => {
    await deleteVideo("vid-9");
    expect(calls[0]).toMatchObject({
      url: `${BASE}/api/videos/vid-9`,
      method: "DELETE",
    });
  });

  it("uploadVideo POSTs multipart form data to /api/videos", async () => {
    responder = () => ({ video_id: "vid-1", job_id: "job-1" });
    const file = new File(["clip"], "clean.mov", { type: "video/quicktime" });
    const res = await uploadVideo({
      file,
      movement: "power_clean",
      loadKg: 102.5,
      recalibrate: true,
    });
    expect(res).toEqual({ video_id: "vid-1", job_id: "job-1" });
    expect(calls[0].url).toBe(`${BASE}/api/videos`);
    expect(calls[0].method).toBe("POST");
    const form = calls[0].body as FormData;
    expect(form).toBeInstanceOf(FormData);
    expect(form.get("file")).toBe(file);
    expect(form.get("movement")).toBe("power_clean");
    expect(form.get("load_kg")).toBe("102.5");
    expect(form.get("recalibrate")).toBe("true");
  });

  it("surfaces a 422 invalid-movement detail as ApiError", async () => {
    responder = () => ({ __status: 422, detail: "unknown movement: yoga" });
    const file = new File(["clip"], "clip.mov", { type: "video/quicktime" });
    const attempt = uploadVideo({
      file,
      movement: "yoga",
      loadKg: 60,
      recalibrate: false,
    });
    await expect(attempt).rejects.toThrowError(ApiError);
    await expect(
      uploadVideo({ file, movement: "yoga", loadKg: 60, recalibrate: false }),
    ).rejects.toMatchObject({ status: 422, message: "unknown movement: yoga" });
  });
});
