"use client";

import type { OverlayRep } from "@/lib/overlay";

function formatScore(score: number): string {
  return Number.isInteger(score) ? String(score) : score.toFixed(1);
}

function MadeChip({ rep }: { rep: OverlayRep }) {
  if (rep.unanalyzed_reason !== null) {
    return (
      <span className="pp-cut-sm inline-flex items-center border border-warn/60 bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-warn">
        Unanalyzed
      </span>
    );
  }
  return rep.made ? (
    <span className="pp-cut-sm inline-flex items-center border border-ok/60 bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-ok">
      Made
    </span>
  ) : (
    <span className="pp-cut-sm inline-flex items-center border border-fail/60 bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-fail">
      Missed
    </span>
  );
}

/**
 * Bottom filmstrip: one cut-corner card per rep. Clicking a card (or using
 * the prev/next controls) selects that rep — the player seeks the video to
 * the rep's t_start.
 */
export function RepFilmstrip({
  reps,
  selected,
  onSelect,
}: {
  reps: OverlayRep[];
  selected: number;
  onSelect: (index: number) => void;
}) {
  return (
    <section aria-label="Reps" className="mt-6">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-display text-lg font-bold uppercase tracking-[0.08em]">
          Reps
        </h3>
        <div className="flex gap-2">
          <button
            type="button"
            aria-label="Previous rep"
            disabled={selected <= 0}
            onClick={() => onSelect(selected - 1)}
            className="pp-cut-sm border border-line px-3 py-1 font-display text-xs font-semibold uppercase tracking-[0.14em] text-ink transition-colors hover:border-accent/40 disabled:cursor-not-allowed disabled:opacity-40"
          >
            &lsaquo; Prev
          </button>
          <button
            type="button"
            aria-label="Next rep"
            disabled={selected >= reps.length - 1}
            onClick={() => onSelect(selected + 1)}
            className="pp-cut-sm border border-line px-3 py-1 font-display text-xs font-semibold uppercase tracking-[0.14em] text-ink transition-colors hover:border-accent/40 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next &rsaquo;
          </button>
        </div>
      </div>

      <ul className="flex gap-3 overflow-x-auto pb-1">
        {reps.map((rep, index) => (
          <li key={rep.rep_index} className="w-40 shrink-0">
            <button
              type="button"
              data-testid="rep-card"
              aria-pressed={index === selected}
              aria-label={`Rep ${rep.rep_index + 1}`}
              onClick={() => onSelect(index)}
              className={`pp-cut pp-card flex h-full w-full flex-col border bg-surface p-3 text-left ${
                index === selected
                  ? "border-accent"
                  : "border-line hover:border-accent/40"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
                  Rep {rep.rep_index + 1}
                </span>
                <MadeChip rep={rep} />
              </div>
              <span
                className={`mt-2 font-display text-3xl font-bold leading-none ${
                  rep.score != null ? "text-accent" : "text-muted"
                }`}
              >
                {rep.score != null ? formatScore(rep.score) : "—"}
              </span>
              <span className="mt-2 font-mono text-[10px] text-muted">
                {rep.t_start.toFixed(2)}s &ndash; {rep.t_end.toFixed(2)}s
              </span>
              {rep.unanalyzed_reason !== null && (
                <span
                  title={rep.unanalyzed_reason}
                  className="mt-1 line-clamp-2 font-mono text-[10px] text-warn"
                >
                  {rep.unanalyzed_reason}
                </span>
              )}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
