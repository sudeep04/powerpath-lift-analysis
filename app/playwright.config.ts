import { defineConfig, devices } from "@playwright/test";

/**
 * Hermetic E2E: the spec mocks the three engine endpoints (overlay, analysis,
 * file) with page.route, so only the Next.js dev server is required — no
 * FastAPI process. See e2e/player.spec.ts for the rationale.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  reporter: [["list"]],
  use: {
    // "localhost" (not 127.0.0.1): the Next dev server treats localhost as
    // its origin and blocks cross-origin dev resources (chunks, HMR) from
    // other hosts, which breaks hydration.
    baseURL: "http://localhost:3211",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev -- --port 3211",
    url: "http://localhost:3211",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
