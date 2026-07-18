"use client";

import type { AnalysisRep, Fault, OverlayRep } from "@/lib/overlay";

function formatScore(score: number): string {
  return Number.isInteger(score) ? String(score) : score.toFixed(1);
}

function formatNumber(value: number, digits = 2): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

const METRIC_ROWS: ReadonlyArray<{
  key: keyof Pick<
    NonNullable<AnalysisRep["metrics"]>,
    | "bar_drift_cm"
    | "peak_concentric_velocity_ms"
    | "path_length_ratio"
    | "smoothness"
  >;
  label: string;
  unit: string;
}> = [
  { key: "bar_drift_cm", label: "Bar drift", unit: "cm" },
  { key: "peak_concentric_velocity_ms", label: "Peak velocity", unit: "m/s" },
  { key: "path_length_ratio", label: "Path ratio", unit: "" },
  { key: "smoothness", label: "Smoothness", unit: "" },
];

const ANGLE_GROUPS: ReadonlyArray<{
  key: "hip_angle_at_phase" | "knee_angle_at_phase" | "elbow_angle_at_phase";
  label: string;
}> = [
  { key: "hip_angle_at_phase", label: "Hip" },
  { key: "knee_angle_at_phase", label: "Knee" },
  { key: "elbow_angle_at_phase", label: "Elbow" },
];

/**
 * Fault tone: "informational" severity is muted; real faults are warn on
 * made reps and fail on missed reps.
 */
function faultTone(fault: Fault, made: boolean): { text: string; border: string } {
  if (fault.severity === "informational") {
    return { text: "text-muted", border: "border-line" };
  }
  return made
    ? { text: "text-warn", border: "border-warn/50" }
    : { text: "text-fail", border: "border-fail/50" };
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h4 className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted">
      {children}
    </h4>
  );
}

/**
 * Right-hand panel for the selected rep: score header, metrics (from
 * metrics.json when available), faults, phases (click to seek), and the
 * M3 "vs your best" placeholder.
 */
export function MetricsPanel({
  rep,
  analysisRep,
  onSeek,
}: {
  rep: OverlayRep;
  analysisRep: AnalysisRep | null;
  onSeek: (t: number) => void;
}) {
  // Faults come from the OVERLAY rep, not the metrics doc: the engine
  // serializes the identical FaultFinding list into both files, but only the
  // overlay shape carries `severity` (metrics.json faults are the 5-key
  // shape). Sourcing from `analysisRep.faults` would drop severity and render
  // informational faults (e.g. catch_above_parallel) as real warn/fail faults.
  const faults = rep.faults;
  const phases = Object.entries(analysisRep?.phases ?? rep.phases).sort(
    (a, b) => a[1] - b[1],
  );
  const metrics = analysisRep?.metrics ?? null;

  return (
    <aside className="flex flex-col gap-4" aria-label="Rep metrics">
      {/* Header: rep number, verdict, score */}
      <div className="pp-cut border border-line bg-surface p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="font-display text-xl font-bold uppercase leading-tight tracking-[0.06em]">
              Rep {rep.rep_index + 1}
            </h3>
            <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
              {rep.t_start.toFixed(2)}s &ndash; {rep.t_end.toFixed(2)}s ·{" "}
              {rep.made ? "made" : "missed"}
            </p>
          </div>
          <div className="text-right">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
              Score
            </p>
            <p
              className={`font-display text-4xl font-bold leading-none ${
                rep.score != null ? "text-accent" : "text-muted"
              }`}
            >
              {rep.score != null ? formatScore(rep.score) : "—"}
            </p>
          </div>
        </div>
        {rep.unanalyzed_reason !== null && (
          <p className="mt-3 border-t border-line pt-3 font-mono text-xs text-warn">
            Unanalyzed: {rep.unanalyzed_reason}
          </p>
        )}
      </div>

      {/* Metrics */}
      <div className="pp-cut border border-line bg-surface p-4">
        <SectionTitle>Metrics</SectionTitle>
        {metrics ? (
          <>
            <dl className="mt-3 grid grid-cols-2 gap-3">
              {METRIC_ROWS.map(({ key, label, unit }) => {
                const value = metrics[key];
                return (
                  <div key={key}>
                    <dt className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted">
                      {label}
                    </dt>
                    <dd className="font-display text-xl font-bold leading-tight">
                      {value != null ? (
                        <>
                          {formatNumber(value)}
                          {unit && (
                            <span className="ml-1 font-mono text-[10px] font-normal uppercase text-muted">
                              {unit}
                            </span>
                          )}
                        </>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </dd>
                  </div>
                );
              })}
            </dl>
            {ANGLE_GROUPS.some(
              (g) => Object.keys(metrics[g.key] ?? {}).length > 0,
            ) && (
              <div className="mt-4 border-t border-line pt-3">
                <SectionTitle>Joint angles</SectionTitle>
                <ul className="mt-2 space-y-1">
                  {ANGLE_GROUPS.flatMap(({ key, label }) =>
                    Object.entries(metrics[key] ?? {}).map(
                      ([phase, angle]) => (
                        <li
                          key={`${key}-${phase}`}
                          className="flex justify-between font-mono text-xs"
                        >
                          <span className="text-muted">
                            {label} @ {phase.replaceAll("_", " ")}
                          </span>
                          <span>{formatNumber(angle, 1)}&deg;</span>
                        </li>
                      ),
                    ),
                  )}
                </ul>
              </div>
            )}
          </>
        ) : (
          <p className="mt-3 font-mono text-xs text-muted">
            {rep.unanalyzed_reason !== null
              ? "No metrics for unanalyzed reps."
              : "Metrics unavailable — engine analysis could not be loaded."}
          </p>
        )}
      </div>

      {/* Faults */}
      <div className="pp-cut border border-line bg-surface p-4">
        <SectionTitle>Faults</SectionTitle>
        {faults.length === 0 ? (
          <p className="mt-3 font-mono text-xs text-muted">
            No faults detected.
          </p>
        ) : (
          <ul className="mt-3 space-y-2">
            {faults.map((fault, index) => {
              const tone = faultTone(fault, rep.made);
              return (
                <li
                  key={`${fault.code}-${fault.phase}-${index}`}
                  className={`pp-cut-sm border ${tone.border} bg-surface-2 p-2.5`}
                >
                  <p className={`text-sm ${tone.text}`}>{fault.message}</p>
                  <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.1em] text-muted">
                    {fault.code} · {fault.phase.replaceAll("_", " ")}
                    {fault.value != null && fault.threshold != null && (
                      <>
                        {" "}
                        · {formatNumber(fault.value)} /{" "}
                        {formatNumber(fault.threshold)}
                      </>
                    )}
                  </p>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Phases */}
      <div className="pp-cut border border-line bg-surface p-4">
        <SectionTitle>Phases</SectionTitle>
        {phases.length === 0 ? (
          <p className="mt-3 font-mono text-xs text-muted">
            No phases detected.
          </p>
        ) : (
          <ul className="mt-3 space-y-1">
            {phases.map(([name, t]) => (
              <li key={name}>
                <button
                  type="button"
                  onClick={() => onSeek(t)}
                  className="flex w-full justify-between font-mono text-xs transition-colors hover:text-accent"
                >
                  <span className="uppercase tracking-[0.1em] text-muted">
                    {name.replaceAll("_", " ")}
                  </span>
                  <span>{t.toFixed(2)}s</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Vs your best — M3 placeholder */}
      <div className="pp-cut pp-blueprint border border-line bg-surface p-4">
        <div className="flex items-center justify-between">
          <SectionTitle>Vs your best</SectionTitle>
          <span className="pp-cut-sm border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
            M3
          </span>
        </div>
        <p className="mt-3 font-mono text-xs text-muted">
          Template comparison against your best rep lands in M3.
        </p>
      </div>
    </aside>
  );
}
