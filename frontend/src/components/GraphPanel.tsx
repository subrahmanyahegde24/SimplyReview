"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import type { EntityType, GraphData, GraphNode } from "@/lib/api";

const GraphCanvas = dynamic(() => import("./GraphCanvas"), { ssr: false });

const LEGEND: EntityType[] = ["Method", "Model", "Dataset", "Metric", "Task", "Concept", "Tool"];

interface Props {
  library: GraphData | null;
  answer: GraphData | null;
}

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, size] as const;
}

export function GraphPanel({ library, answer }: Props) {
  const [view, setView] = useState<"library" | "answer">("library");
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [frameRef, size] = useSize<HTMLDivElement>();

  // Jump to the answer's subgraph as soon as the graph search returns it.
  const [shownAnswer, setShownAnswer] = useState(answer);
  if (answer !== shownAnswer) {
    setShownAnswer(answer);
    setView(answer ? "answer" : "library");
    setSelected(null);
  }

  const data = view === "answer" && answer ? answer : library;
  const relations = selected && data ? data.links.filter((l) => l.source === selected.id || l.target === selected.id) : [];
  const nameOf = (id: string) => data?.nodes.find((n) => n.id === id)?.name ?? id;

  return (
    <aside className="graph-panel" aria-label="Knowledge graph">
      <header className="panel-head">
        <h2 className="section-title">Knowledge graph</h2>
        <div className="segmented" role="tablist" aria-label="Graph view">
          <button role="tab" aria-selected={view === "library"} onClick={() => setView("library")}>
            Whole library
          </button>
          <button role="tab" aria-selected={view === "answer"} disabled={!answer} onClick={() => setView("answer")}>
            This answer
          </button>
        </div>
      </header>

      <div className="graph-frame" ref={frameRef}>
        {data && data.nodes.length > 0 && size.width > 0 ? (
          <GraphCanvas
            data={data}
            width={size.width}
            height={size.height}
            selected={selected?.id ?? null}
            onSelect={setSelected}
          />
        ) : (
          <p className="empty">Entities and their relations appear here once a paper is added.</p>
        )}
      </div>

      {selected ? (
        <div className="entity-card">
          <p className="entity-name">
            <span className={`type-dot type-${selected.type.toLowerCase()}`} aria-hidden />
            {selected.name}
            <span className="entity-type">{selected.type}</span>
          </p>
          {selected.description && <p className="entity-desc">{selected.description}</p>}
          {relations.length > 0 && (
            <ul className="entity-relations">
              {relations.slice(0, 6).map((l, i) => (
                <li key={i}>
                  {nameOf(l.source)} <em>{l.type.replaceAll("_", " ").toLowerCase()}</em> {nameOf(l.target)}
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <ul className="legend" aria-label="Entity types">
          {LEGEND.map((t) => (
            <li key={t}>
              <span className={`type-dot type-${t.toLowerCase()}`} aria-hidden />
              {t}
            </li>
          ))}
          {view === "answer" && (
            <li>
              <span className="type-dot seed-dot" aria-hidden />
              Matched to your question
            </li>
          )}
        </ul>
      )}
    </aside>
  );
}
