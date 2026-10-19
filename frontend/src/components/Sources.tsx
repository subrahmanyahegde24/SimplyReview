"use client";

import { useEffect, useRef } from "react";
import type { ChunkHit, Source } from "@/lib/api";

const ORIGIN_LABEL = { vector: "Search", graph: "Graph", both: "Search and graph" } as const;

interface Props {
  sources: Source[];
  hits: ChunkHit[];
  active: number | null;
}

export function Sources({ sources, hits, active }: Props) {
  const refs = useRef(new Map<number, HTMLLIElement>());

  useEffect(() => {
    if (active === null) return;
    refs.current.get(active)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [active]);

  if (!sources.length) return null;
  const hitById = new Map(hits.map((h) => [h.chunk_id, h]));

  return (
    <section className="sources" aria-label="Sources">
      <h2 className="section-title">Sources</h2>
      <ol>
        {sources.map((source) => {
          const hit = hitById.get(source.chunk_id);
          return (
            <li
              key={source.chunk_id}
              ref={(el) => {
                if (el) refs.current.set(source.n, el);
                else refs.current.delete(source.n);
              }}
              className={`source origin-${source.origin}`}
              data-active={active === source.n}
            >
              <details open={active === source.n}>
                <summary>
                  <span className="source-n">{source.n}</span>
                  <span className="source-title">{source.paper_title}</span>
                  <span className={`origin-tag origin-${source.origin}`}>{ORIGIN_LABEL[source.origin]}</span>
                  <span className="source-page">p. {source.page}</span>
                </summary>
                <div className="source-meta">
                  {hit?.dense_rank && <span>Meaning rank {hit.dense_rank}</span>}
                  {hit?.keyword_rank && <span>Keyword rank {hit.keyword_rank}</span>}
                </div>
                <p className="source-text">
                  {source.text}
                  {source.text.length >= 600 ? "…" : ""}
                </p>
              </details>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
