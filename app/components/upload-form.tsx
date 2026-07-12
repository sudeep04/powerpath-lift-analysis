"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { listMovements, uploadVideo, type Movement } from "@/lib/api";
import { rememberJob } from "@/lib/active-jobs";

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function UploadForm() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [movements, setMovements] = useState<Movement[]>([]);
  const [movementsError, setMovementsError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [movement, setMovement] = useState("");
  const [loadKgText, setLoadKgText] = useState("");
  const [recalibrate, setRecalibrate] = useState(false);

  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listMovements()
      .then((list) => {
        if (!cancelled) setMovements(list);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setMovementsError(
            err instanceof Error ? err.message : "Failed to load movements",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const families = useMemo(() => {
    const grouped = new Map<string, Movement[]>();
    for (const m of movements) {
      const list = grouped.get(m.family) ?? [];
      list.push(m);
      grouped.set(m.family, list);
    }
    return [...grouped.entries()];
  }, [movements]);

  const loadKg = Number(loadKgText);
  const valid =
    file !== null && movement !== "" && loadKgText !== "" && loadKg > 0;

  const pickFile = (picked: File | null | undefined) => {
    if (picked) setFile(picked);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!valid || !file || submitting) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await uploadVideo({ file, movement, loadKg, recalibrate });
      rememberJob(res.video_id, res.job_id);
      router.push("/");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Upload failed");
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mx-auto max-w-xl">
      <h2 className="font-display text-2xl font-bold uppercase tracking-[0.06em]">
        Upload a lift
      </h2>
      <p className="mt-1 font-mono text-xs uppercase tracking-[0.1em] text-muted">
        One athlete · side-on · plate marker visible
      </p>

      {/* Dropzone */}
      <div
        data-testid="dropzone"
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pickFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
        role="button"
        tabIndex={0}
        aria-label="Choose a video file"
        className={`pp-cut mt-6 flex min-h-44 cursor-pointer flex-col items-center justify-center border px-6 py-10 text-center transition-colors ${
          dragging
            ? "border-accent bg-surface-2"
            : "border-line bg-surface hover:border-accent/40"
        }`}
      >
        <svg
          aria-hidden="true"
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          className={dragging ? "text-accent" : "text-muted"}
        >
          <path d="M12 16V4m0 0-4 4m4-4 4 4" />
          <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
        </svg>
        {file ? (
          <>
            <p className="mt-3 font-mono text-sm text-ink">{file.name}</p>
            <p className="mt-1 font-mono text-xs text-muted">
              {formatSize(file.size)} — click to replace
            </p>
          </>
        ) : (
          <>
            <p className="mt-3 font-display text-lg font-semibold uppercase tracking-[0.08em]">
              Drop video here
            </p>
            <p className="mt-1 font-mono text-xs text-muted">
              or click to browse — .mov / .mp4
            </p>
          </>
        )}
        <input
          ref={fileInputRef}
          data-testid="file-input"
          type="file"
          accept="video/*"
          className="hidden"
          onClick={(e) => e.stopPropagation()}
          onChange={(e) => pickFile(e.target.files?.[0])}
        />
      </div>

      {/* Movement */}
      <div className="mt-6">
        <label
          htmlFor="movement"
          className="mb-1.5 block font-mono text-[11px] uppercase tracking-[0.14em] text-muted"
        >
          Movement
        </label>
        <select
          id="movement"
          value={movement}
          onChange={(e) => setMovement(e.target.value)}
          className="w-full border border-line bg-surface px-3 py-2.5 font-body text-sm text-ink focus:border-accent focus:outline-none"
        >
          <option value="">Select movement…</option>
          {families.map(([family, list]) => (
            <optgroup key={family} label={titleCase(family)}>
              {list.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.display_name}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        {movementsError && (
          <p className="mt-1.5 font-mono text-xs text-fail">{movementsError}</p>
        )}
      </div>

      {/* Load */}
      <div className="mt-5">
        <label
          htmlFor="load-kg"
          className="mb-1.5 block font-mono text-[11px] uppercase tracking-[0.14em] text-muted"
        >
          Load
        </label>
        <div className="flex items-stretch border border-line bg-surface focus-within:border-accent">
          <input
            id="load-kg"
            type="number"
            inputMode="decimal"
            min="0"
            step="0.5"
            placeholder="0.0"
            value={loadKgText}
            onChange={(e) => setLoadKgText(e.target.value)}
            className="w-full bg-transparent px-3 py-2.5 font-mono text-sm text-ink placeholder:text-muted/60 focus:outline-none"
          />
          <span className="flex items-center border-l border-line px-3 font-mono text-xs uppercase tracking-[0.12em] text-muted">
            kg
          </span>
        </div>
      </div>

      {/* Recalibrate */}
      <div className="mt-5 flex items-center justify-between border border-line bg-surface px-3 py-2.5">
        <div>
          <p className="font-body text-sm text-ink">Recalibrate</p>
          <p className="font-mono text-xs text-muted">
            Re-measure plate scale for this video
          </p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={recalibrate}
          aria-label="Recalibrate"
          onClick={() => setRecalibrate((v) => !v)}
          className={`pp-cut-sm relative h-6 w-11 shrink-0 border transition-colors ${
            recalibrate ? "border-accent bg-accent/20" : "border-line bg-surface-2"
          }`}
        >
          <span
            className={`absolute top-1 h-3.5 w-3.5 transition-[left] ${
              recalibrate ? "left-6 bg-accent" : "left-1 bg-muted"
            }`}
          />
        </button>
      </div>

      {submitError && (
        <p className="mt-4 font-mono text-xs text-fail">{submitError}</p>
      )}

      <button
        type="submit"
        disabled={!valid || submitting}
        className="pp-cut mt-6 w-full bg-accent px-6 py-3 font-display text-sm font-bold uppercase tracking-[0.14em] text-bg transition-[filter] hover:brightness-110 disabled:cursor-not-allowed disabled:bg-surface-2 disabled:text-muted"
      >
        {submitting ? "Uploading…" : "Upload & analyze"}
      </button>
    </form>
  );
}
