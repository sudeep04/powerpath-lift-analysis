import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UploadForm } from "@/components/upload-form";

const push = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

const MOVEMENTS = [
  { key: "power_clean", display_name: "Power Clean", family: "clean" },
  { key: "back_squat", display_name: "Back Squat", family: "squat" },
  { key: "push_press", display_name: "Push Press", family: "press" },
  { key: "power_snatch", display_name: "Power Snatch", family: "snatch" },
  { key: "deadlift", display_name: "Deadlift", family: "deadlift" },
];

function jsonResponse(data: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    json: async () => data,
  } as unknown as Response;
}

let uploads: FormData[];

beforeEach(() => {
  push.mockClear();
  uploads = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = init?.method ?? "GET";
      if (url.endsWith("/api/movements") && method === "GET") {
        return jsonResponse(MOVEMENTS);
      }
      if (url.endsWith("/api/videos") && method === "POST") {
        uploads.push(init?.body as FormData);
        return jsonResponse({ video_id: "vid-1", job_id: "job-1" });
      }
      return jsonResponse({ detail: "unexpected request" }, 500);
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  window.sessionStorage.clear();
});

async function renderForm() {
  render(<UploadForm />);
  // movements loaded → grouped option appears
  await screen.findByRole("option", { name: "Power Clean" });
  return {
    submit: screen.getByRole("button", { name: /upload & analyze/i }),
    fileInput: screen.getByTestId("file-input") as HTMLInputElement,
    movementSelect: screen.getByLabelText(/movement/i),
    loadInput: screen.getByLabelText(/load/i),
  };
}

const videoFile = () =>
  new File(["clip-bytes"], "clean.mov", { type: "video/quicktime" });

describe("upload form validation", () => {
  it("groups movements by family in the select", async () => {
    await renderForm();
    expect(screen.getByRole("group", { name: "Clean" })).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Squat" })).toBeInTheDocument();
    expect(screen.getAllByRole("option")).toHaveLength(MOVEMENTS.length + 1);
  });

  it("disables submit with no file even when movement and load are set", async () => {
    const user = userEvent.setup();
    const { submit, movementSelect, loadInput } = await renderForm();
    expect(submit).toBeDisabled();
    await user.selectOptions(movementSelect, "power_clean");
    await user.type(loadInput, "100");
    expect(submit).toBeDisabled();
  });

  it("disables submit with no movement selected", async () => {
    const user = userEvent.setup();
    const { submit, fileInput, loadInput } = await renderForm();
    await user.upload(fileInput, videoFile());
    await user.type(loadInput, "100");
    expect(submit).toBeDisabled();
  });

  it("disables submit when load is missing, zero, or negative", async () => {
    const user = userEvent.setup();
    const { submit, fileInput, movementSelect, loadInput } = await renderForm();
    await user.upload(fileInput, videoFile());
    await user.selectOptions(movementSelect, "power_clean");
    expect(submit).toBeDisabled(); // load empty

    await user.type(loadInput, "0");
    expect(submit).toBeDisabled(); // load = 0

    await user.clear(loadInput);
    await user.type(loadInput, "-20");
    expect(submit).toBeDisabled(); // load < 0
  });

  it("enables submit once file, movement, and positive load are set", async () => {
    const user = userEvent.setup();
    const { submit, fileInput, movementSelect, loadInput } = await renderForm();
    await user.upload(fileInput, videoFile());
    await user.selectOptions(movementSelect, "power_clean");
    await user.type(loadInput, "102.5");
    expect(submit).toBeEnabled();
  });

  it("submits the upload and navigates to the library", async () => {
    const user = userEvent.setup();
    const { submit, fileInput, movementSelect, loadInput } = await renderForm();
    await user.upload(fileInput, videoFile());
    await user.selectOptions(movementSelect, "power_clean");
    await user.type(loadInput, "102.5");
    await user.click(submit);

    await waitFor(() => expect(push).toHaveBeenCalledWith("/"));
    expect(uploads).toHaveLength(1);
    expect(uploads[0].get("movement")).toBe("power_clean");
    expect(uploads[0].get("load_kg")).toBe("102.5");
    expect(uploads[0].get("recalibrate")).toBe("false");
    // job id remembered for library polling
    expect(
      window.sessionStorage.getItem("powerpath.active-jobs"),
    ).toContain("job-1");
  });
});
