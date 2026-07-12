import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { VideoSummary } from "@/lib/api";
import { Library } from "@/components/library";

vi.mock("next/link", () => ({
  default: ({
    href,
    children,
    ...props
  }: React.ComponentProps<"a"> & { href: string }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

function video(overrides: Partial<VideoSummary>): VideoSummary {
  return {
    video_id: "vid-1",
    movement: "power_clean",
    display_name: "Power Clean",
    load_kg: 102.5,
    filmed_at: "2026-07-10T18:30:00Z",
    job: { state: "DONE", progress: 100, stage: null, error: null },
    rep_count: 5,
    best_score: 87,
    ...overrides,
  };
}

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    json: async () => data,
    blob: async () => new Blob(["bytes"], { type: "video/quicktime" }),
  } as unknown as Response;
}

type Route = (url: string, method: string, init?: RequestInit) => unknown;

let route: Route;

beforeEach(() => {
  route = () => [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      return jsonResponse(route(url, method, init));
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  window.sessionStorage.clear();
});

describe("library states", () => {
  it("renders the empty-state hero when there are no videos", async () => {
    route = () => [];
    render(<Library />);
    expect(
      await screen.findByText(/drop your first video/i),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /upload a lift/i }),
    ).toHaveAttribute("href", "/upload");
  });

  it("renders a RUNNING card with progress bar and stage label", async () => {
    route = () => [
      video({
        job: { state: "RUNNING", progress: 42, stage: "pose", error: null },
        rep_count: null,
        best_score: null,
      }),
    ];
    render(<Library />);
    expect(await screen.findByText("RUNNING")).toBeInTheDocument();
    const bar = screen.getByRole("progressbar", { name: /analysis progress/i });
    expect(bar).toHaveAttribute("aria-valuenow", "42");
    expect(screen.getByText(/pose · 42%/i)).toBeInTheDocument();
    expect(screen.getByText("Power Clean")).toBeInTheDocument();
    expect(screen.getByText("102.5 KG")).toBeInTheDocument();
  });

  it("renders a FAILED card with the error reason and a Retry that re-POSTs", async () => {
    const failed = video({
      job: {
        state: "FAILED",
        progress: 63,
        stage: "bar",
        error: "bar marker lost during pull",
      },
      rep_count: null,
      best_score: null,
    });
    const requests: { url: string; method: string }[] = [];
    let listCalls = 0;
    route = (url, method) => {
      requests.push({ url, method });
      if (method === "GET" && url.endsWith("/api/videos")) {
        listCalls += 1;
        return listCalls === 1 ? [failed] : [];
      }
      if (method === "GET" && url.endsWith("/api/videos/vid-1/file")) {
        return {}; // blob comes from jsonResponse's blob()
      }
      if (method === "POST" && url.endsWith("/api/videos")) {
        return { video_id: "vid-2", job_id: "job-2" };
      }
      if (method === "DELETE" && url.endsWith("/api/videos/vid-1")) {
        return {};
      }
      return [];
    };

    const user = userEvent.setup();
    render(<Library />);
    expect(
      await screen.findByText("bar marker lost during pull"),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /retry/i }));

    await waitFor(() => {
      expect(requests).toContainEqual({
        url: "http://127.0.0.1:8400/api/videos",
        method: "POST",
      });
    });
    expect(requests).toContainEqual({
      url: "http://127.0.0.1:8400/api/videos/vid-1",
      method: "DELETE",
    });
  });

  it("renders a DONE card with rep count and best score", async () => {
    route = () => [video({ rep_count: 5, best_score: 87 })];
    render(<Library />);
    expect(await screen.findByText("DONE")).toBeInTheDocument();
    expect(screen.getByText("Reps")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("Best score")).toBeInTheDocument();
    expect(screen.getByText("87")).toBeInTheDocument();
    // no progress bar once done
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });
});
