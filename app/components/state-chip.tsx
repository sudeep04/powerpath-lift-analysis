import type { JobState } from "@/lib/api";

const STYLES: Record<JobState, string> = {
  QUEUED: "border-line text-muted",
  RUNNING: "border-accent/60 text-accent pp-pulse",
  DONE: "border-ok/60 text-ok",
  FAILED: "border-fail/60 text-fail",
};

export function StateChip({ state }: { state: JobState }) {
  return (
    <span
      className={`pp-cut-sm inline-flex items-center border bg-surface-2 px-2 py-0.5 font-mono text-[11px] uppercase tracking-[0.14em] ${STYLES[state]}`}
    >
      {state}
    </span>
  );
}
