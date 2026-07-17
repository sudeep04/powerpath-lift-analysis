import { expect, test } from "@playwright/test";
import { ANALYSIS_FIXTURE, OVERLAY_FIXTURE } from "../lib/overlay-fixture";

/**
 * Single happy-path E2E for the analysis player.
 *
 * Approach note: the task allowed either standing up the FastAPI app in
 * POWERPATH_FAKE_ENGINE=1 mode or hermetically mocking the engine endpoints.
 * We mock — page.route serves the shared fixture (frozen contract shape,
 * the same document the Vitest suites render) for overlay/analysis/file, so
 * the test needs no Python process and cannot flake on engine startup.
 */

const VIDEO_ID = "vid-e2e";

test("player renders overlay canvas + filmstrip and rep cards seek the video", async ({
  page,
}) => {
  await page.route(`**/api/videos/${VIDEO_ID}/overlay`, (route) =>
    route.fulfill({ json: OVERLAY_FIXTURE }),
  );
  await page.route(`**/api/videos/${VIDEO_ID}/analysis`, (route) =>
    route.fulfill({ json: ANALYSIS_FIXTURE }),
  );
  await page.route(`**/api/videos/${VIDEO_ID}/file`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "video/mp4",
      body: Buffer.alloc(64),
    }),
  );

  await page.goto(`/video/${VIDEO_ID}`);

  // Video + overlay canvas mount.
  await expect(page.locator("video")).toBeVisible();
  await expect(page.getByTestId("overlay-canvas")).toBeVisible();

  // Filmstrip renders one card per rep (fixture has 5).
  const cards = page.getByTestId("rep-card");
  await expect(cards).toHaveCount(OVERLAY_FIXTURE.reps.length);

  // Metrics panel defaults to rep 1 with the M3 placeholder present.
  await expect(page.getByRole("heading", { name: /rep 1/i })).toBeVisible();
  await expect(page.getByText("M3", { exact: true })).toBeVisible();

  // Clicking a card seeks the video to that rep's t_start and selects it.
  await cards.nth(2).click();
  await expect(cards.nth(2)).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("heading", { name: /rep 3/i })).toBeVisible();
  const currentTime = await page
    .locator("video")
    .evaluate((el) => (el as HTMLVideoElement).currentTime);
  expect(currentTime).toBeCloseTo(OVERLAY_FIXTURE.reps[2].t_start, 3);

  // Unanalyzed rep surfaces its reason on the filmstrip.
  await expect(page.getByText("bar marker lost during catch")).toBeVisible();
});
