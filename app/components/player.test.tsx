import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ANALYSIS_FIXTURE, OVERLAY_FIXTURE } from "@/lib/overlay-fixture";
import { Player } from "@/components/player";

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    json: async () => data,
  } as unknown as Response;
}

beforeEach(() => {
  // Route the two data fetches; anything else is unexpected.
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/videos/vid-1/overlay")) {
        return jsonResponse(OVERLAY_FIXTURE);
      }
      if (url.endsWith("/api/videos/vid-1/analysis")) {
        return jsonResponse(ANALYSIS_FIXTURE);
      }
      return jsonResponse({ detail: "not found" }, 404);
    }),
  );
  // jsdom lacks matchMedia; report reduced motion so the player uses static
  // event-driven draws instead of an endless rAF loop.
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({
      matches: query.includes("prefers-reduced-motion"),
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })),
  );
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  // jsdom has no 2D canvas; return a permissive stub so the draw path runs.
  const ctxStub = {
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
    measureText: vi.fn(() => ({ width: 42 })),
    setTransform: vi.fn(),
  } as unknown as CanvasRenderingContext2D;
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctxStub);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Player", () => {
  it("renders video, overlay canvas, filmstrip, and metrics for rep 1", async () => {
    const { container } = render(<Player videoId="vid-1" />);

    expect(await screen.findAllByTestId("rep-card")).toHaveLength(5);
    expect(screen.getByTestId("overlay-canvas")).toBeInTheDocument();

    const video = container.querySelector("video");
    expect(video).not.toBeNull();
    expect(video?.getAttribute("src")).toBe(
      "http://127.0.0.1:8400/api/videos/vid-1/file",
    );

    // Metrics panel defaults to the first rep.
    expect(
      screen.getByRole("heading", { name: /rep 1/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("Bar drift")).toBeInTheDocument();
    // "Vs your best" placeholder is labeled M3.
    expect(screen.getByText(/vs your best/i)).toBeInTheDocument();
    expect(screen.getByText("M3")).toBeInTheDocument();
  });

  it("clicking a filmstrip card seeks the video to that rep's t_start", async () => {
    const user = userEvent.setup();
    const { container } = render(<Player videoId="vid-1" />);

    const cards = await screen.findAllByTestId("rep-card");
    await user.click(cards[2]);

    const video = container.querySelector("video") as HTMLVideoElement;
    expect(video.currentTime).toBeCloseTo(OVERLAY_FIXTURE.reps[2].t_start, 5);

    // Metrics panel follows the selection.
    expect(
      screen.getByRole("heading", { name: /rep 3/i }),
    ).toBeInTheDocument();
    expect(cards[2]).toHaveAttribute("aria-pressed", "true");
  });

  it("prev/next buttons seek to the neighboring rep", async () => {
    const user = userEvent.setup();
    const { container } = render(<Player videoId="vid-1" />);
    await screen.findAllByTestId("rep-card");

    await user.click(screen.getByRole("button", { name: /next rep/i }));
    const video = container.querySelector("video") as HTMLVideoElement;
    expect(video.currentTime).toBeCloseTo(OVERLAY_FIXTURE.reps[1].t_start, 5);

    await user.click(screen.getByRole("button", { name: /previous rep/i }));
    expect(video.currentTime).toBeCloseTo(OVERLAY_FIXTURE.reps[0].t_start, 5);
  });

  it("shows the error card when the overlay fetch fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ detail: "video not found" }, 404)),
    );
    render(<Player videoId="vid-1" />);
    expect(await screen.findByText(/analysis unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/video not found/i)).toBeInTheDocument();
  });

  it("still renders metrics from the overlay when metrics.json fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/overlay")) return jsonResponse(OVERLAY_FIXTURE);
        return jsonResponse({ detail: "boom" }, 500);
      }),
    );
    render(<Player videoId="vid-1" />);

    expect(await screen.findAllByTestId("rep-card")).toHaveLength(5);
    expect(screen.getByText(/metrics unavailable/i)).toBeInTheDocument();
    // Faults still come from overlay.json.
    expect(
      screen.getByText(/bar drifts 4\.2cm forward at the knee/i),
    ).toBeInTheDocument();
  });
});
