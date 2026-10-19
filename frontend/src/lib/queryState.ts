import type { ChunkHit, GraphData, QueryEvent, Relation, Seed, Source, StageName } from "./api";

export interface StageTiming {
  start: number; // ms since the question was sent
  end?: number;
  error?: string;
}

export interface QueryState {
  question: string;
  status: "idle" | "running" | "done" | "error";
  startedAt: number;
  stages: Partial<Record<StageName | "combine", StageTiming>>;
  hits: ChunkHit[];
  seeds: Seed[];
  relations: Relation[];
  subgraph: GraphData | null;
  sources: Source[];
  answer: string;
  firstTokenMs?: number;
  totalMs?: number;
  error?: string;
}

export const initialQuery: QueryState = {
  question: "",
  status: "idle",
  startedAt: 0,
  stages: {},
  hits: [],
  seeds: [],
  relations: [],
  subgraph: null,
  sources: [],
  answer: "",
};

export type QueryAction =
  | { type: "start"; question: string; at: number }
  | { type: "event"; event: QueryEvent; at: number }
  | { type: "fail"; message: string };

export function queryReducer(state: QueryState, action: QueryAction): QueryState {
  switch (action.type) {
    case "start":
      return { ...initialQuery, question: action.question, status: "running", startedAt: action.at };
    case "fail":
      return { ...state, status: "error", error: action.message };
    case "event":
      return applyEvent(state, action.event, action.at - state.startedAt);
  }
}

function applyEvent(state: QueryState, event: QueryEvent, t: number): QueryState {
  switch (event.type) {
    case "stage": {
      const previous = state.stages[event.stage];
      const timing: StageTiming =
        event.status === "running"
          ? { start: t }
          : { start: previous?.start ?? t, end: t, error: event.error };
      return {
        ...state,
        stages: { ...state.stages, [event.stage]: timing },
        firstTokenMs: event.first_token_ms ?? state.firstTokenMs,
      };
    }
    case "vector":
      return { ...state, hits: event.hits };
    case "graph":
      return { ...state, seeds: event.seeds, relations: event.relations, subgraph: event.subgraph };
    case "sources":
      return { ...state, sources: event.sources, stages: { ...state.stages, combine: { start: t, end: t } } };
    case "token":
      return { ...state, answer: state.answer + event.text };
    case "done":
      return { ...state, status: "done", totalMs: event.timings.total };
    case "error":
      return { ...state, status: "error", error: event.message };
  }
}
