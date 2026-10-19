"use client";

import { useEffect, useState } from "react";
import type { QueryState, StageTiming } from "@/lib/queryState";

interface Row {
  key: keyof QueryState["stages"];
  label: string;
  lane: "neutral" | "vector" | "graph" | "answer";
  detail: (s: QueryState) => string;
}

const ROWS: Row[] = [
  { key: "embed", label: "Understand the question", lane: "neutral", detail: () => "Gemini embedding" },
  {
    key: "vector",
    label: "Keyword + meaning search",
    lane: "vector",
    detail: (s) => (s.hits.length ? `${s.hits.length} passages from Qdrant, fused with RRF` : "Qdrant BM25 and dense vectors"),
  },
  {
    key: "graph",
    label: "Knowledge graph",
    lane: "graph",
    detail: (s) =>
      s.seeds.length
        ? `${s.seeds.length} entities matched, ${s.relations.length} relations followed in Neo4j`
        : "Neo4j entity links and relations",
  },
  {
    key: "combine",
    label: "Combine evidence",
    lane: "neutral",
    detail: (s) => {
      if (!s.sources.length) return "Deduplicate and number sources";
      const count = (o: string) => s.sources.filter((x) => x.origin === o).length;
      return `${s.sources.length} sources: ${count("vector")} search, ${count("graph")} graph, ${count("both")} both`;
    },
  },
  {
    key: "generate",
    label: "Write the answer",
    lane: "answer",
    detail: (s) => (s.firstTokenMs ? `First words after ${formatMs(s.firstTokenMs)}` : "Gemini, citing every claim"),
  },
];

export function formatMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function useNow(running: boolean, startedAt: number): number {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (!running) return;
    let frame = requestAnimationFrame(function tick() {
      setNow(performance.now() - startedAt);
      frame = requestAnimationFrame(tick);
    });
    return () => cancelAnimationFrame(frame);
  }, [running, startedAt]);
  return now;
}

export function Trace({ state }: { state: QueryState }) {
  const running = state.status === "running";
  const now = useNow(running, state.startedAt);
  const ends = Object.values(state.stages).map((s) => (s as StageTiming).end ?? now);
  const span = Math.max(running ? now : 0, ...ends, 1);

  return (
    <section className="trace" aria-label="Retrieval trace" aria-live="polite">
      <ol className="trace-rows">
        {ROWS.map((row) => {
          const timing = state.stages[row.key];
          const done = timing?.end !== undefined;
          const end = timing ? (timing.end ?? now) : 0;
          const left = timing ? (timing.start / span) * 100 : 0;
          const width = timing ? Math.max(((end - timing.start) / span) * 100, 0.6) : 0;
          const status = !timing ? "waiting" : timing.error ? "failed" : done ? "done" : "running";
          return (
            <li key={row.key} className={`trace-row lane-${row.lane}`} data-status={status}>
              <div className="trace-label">
                <span className="trace-name">{row.label}</span>
                <span className="trace-detail">{timing?.error ? `Failed: ${timing.error}` : row.detail(state)}</span>
              </div>
              <div className="trace-track">
                {timing && <span className="trace-bar" style={{ left: `${left}%`, width: `${width}%` }} />}
              </div>
              <span className="trace-time">{timing ? formatMs(end - timing.start) : ""}</span>
            </li>
          );
        })}
      </ol>
      <p className="trace-total">
        {state.status === "done" && state.totalMs !== undefined
          ? `Answered in ${formatMs(state.totalMs)}. Search and graph ran in parallel.`
          : running
            ? `Working… ${formatMs(now)}`
            : ""}
      </p>
    </section>
  );
}
