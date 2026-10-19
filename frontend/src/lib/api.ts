export type EntityType = "Method" | "Model" | "Dataset" | "Metric" | "Task" | "Concept" | "Tool";

export interface Stats {
  papers: number;
  chunks: number;
  entities: number;
  relations: number;
}

export interface Paper {
  id: string;
  title: string;
  authors: string[] | null;
  year: number | null;
  summary: string | null;
  filename: string;
  pages: number;
  entities: number;
}

export interface GraphNode {
  id: string;
  name: string;
  type: EntityType;
  description?: string;
  mentions?: number;
  degree?: number;
  seed?: boolean;
}

export interface GraphLink {
  source: string;
  target: string;
  type: string;
  description: string;
  weight?: number;
  bridge?: boolean;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface ChunkHit {
  chunk_id: string;
  paper_title: string;
  page: number;
  score: number;
  dense_rank: number | null;
  keyword_rank: number | null;
}

export interface Seed {
  key: string;
  name: string;
  type: EntityType;
  score: number;
  via: "semantic" | "keyword" | "both";
}

export interface Relation {
  source_name: string;
  target_name: string;
  type: string;
  description: string;
  papers: number;
  bridge: boolean;
}

export interface Source {
  n: number;
  chunk_id: string;
  paper_id: string;
  paper_title: string;
  page: number;
  text: string;
  origin: "vector" | "graph" | "both";
}

export type StageName = "embed" | "vector" | "graph" | "generate";

export type QueryEvent =
  | { type: "stage"; stage: StageName; status: "running" | "done" | "error"; ms?: number; error?: string; first_token_ms?: number }
  | { type: "vector"; hits: ChunkHit[] }
  | { type: "graph"; seeds: Seed[]; relations: Relation[]; subgraph: GraphData }
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; timings: Record<string, number> }
  | { type: "error"; message: string };

export type IngestEvent =
  | { type: "stage"; stage: "parse" | "embed" | "extract" | "link"; status: "running" | "done"; [key: string]: unknown }
  | { type: "progress"; done: number; total: number; entities: number }
  | { type: "meta"; title: string; year: number | null }
  | { type: "done"; title: string; chunks: number; entities: number; relations: number; seconds: number }
  | { type: "skipped"; reason: string }
  | { type: "error"; message: string };

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} returned ${res.status}`);
  return res.json();
}

/** POST and read a Server-Sent Events stream, calling onEvent for each `data:` message. */
export async function streamEvents<E>(
  path: string,
  init: RequestInit,
  onEvent: (event: E) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(path, { ...init, method: "POST", signal });
  if (!res.ok || !res.body) {
    let detail = "";
    try {
      detail = (await res.json()).detail;
    } catch {}
    throw new Error(detail || `Request failed with status ${res.status}`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (data) onEvent(JSON.parse(data) as E);
    }
  }
}
