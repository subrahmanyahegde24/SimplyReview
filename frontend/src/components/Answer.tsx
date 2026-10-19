"use client";

import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import type { Source } from "@/lib/api";

const CITATION_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/** Turn "[1, 4]" into markdown links the renderer below draws as citation chips. */
function linkCitations(text: string): string {
  return text.replace(CITATION_RE, (_, list: string) =>
    list
      .split(",")
      .map((n) => `[${n.trim()}](#cite-${n.trim()})`)
      .join(""),
  );
}

interface Props {
  text: string;
  streaming: boolean;
  sources: Source[];
  onCite: (n: number) => void;
}

export function Answer({ text, streaming, sources, onCite }: Props) {
  return (
    <article className="answer" data-streaming={streaming}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a({ href, children }) {
            const match = href?.match(/^#cite-(\d+)$/);
            if (!match) return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
            const n = Number(match[1]);
            const source = sources.find((s) => s.n === n);
            return (
              <button
                type="button"
                className={`cite origin-${source?.origin ?? "vector"}`}
                onClick={() => onCite(n)}
                title={source ? `${source.paper_title}, page ${source.page}` : undefined}
              >
                {n}
              </button>
            );
          },
          table({ children }) {
            return (
              <div className="table-scroll">
                <table>{children}</table>
              </div>
            );
          },
        }}
      >
        {linkCitations(text)}
      </ReactMarkdown>
    </article>
  );
}
