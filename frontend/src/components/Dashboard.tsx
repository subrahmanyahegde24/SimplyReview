"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { getJson, streamEvents, type GraphData, type Paper, type QueryEvent, type Stats } from "@/lib/api";
import { initialQuery, queryReducer } from "@/lib/queryState";
import { Answer } from "./Answer";
import { GraphPanel } from "./GraphPanel";
import { Library } from "./Library";
import { Sources } from "./Sources";
import { Trace } from "./Trace";

const SUGGESTIONS = [
  "Which methods are evaluated on which datasets?",
  "How do these papers build on each other?",
  "Compare the reported results across papers.",
];

export function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [library, setLibrary] = useState<GraphData | null>(null);
  const [locked, setLocked] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [state, dispatch] = useReducer(queryReducer, initialQuery);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(
    () =>
      Promise.all([
        getJson<Stats>("/api/stats"),
        getJson<Paper[]>("/api/papers"),
        getJson<GraphData>("/api/graph?limit=150"),
      ]).then(
        ([s, p, g]) => {
          setStats(s);
          setPapers(p);
          setLibrary(g);
          setLoadError(null);
        },
        () => setLoadError("The research service isn't reachable. Check that the backend is running, then reload."),
      ),
    [],
  );

  useEffect(() => {
    void refresh();
    getJson<{ uploads_locked: boolean }>("/api/config")
      .then((c) => setLocked(c.uploads_locked))
      .catch(() => {});
  }, [refresh]);

  async function ask(text: string) {
    const q = text.trim();
    if (q.length < 3 || state.status === "running") return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setActiveSource(null);
    dispatch({ type: "start", question: q, at: performance.now() });
    try {
      await streamEvents<QueryEvent>(
        "/api/query",
        { body: JSON.stringify({ question: q }), headers: { "content-type": "application/json" } },
        (event) => dispatch({ type: "event", event, at: performance.now() }),
        controller.signal,
      );
    } catch (err) {
      if (!controller.signal.aborted) dispatch({ type: "fail", message: err instanceof Error ? err.message : String(err) });
    }
  }

  const running = state.status === "running";
  const hasPapers = (stats?.papers ?? 0) > 0;

  return (
    <div className="app">
      <header className="masthead">
        <h1 className="wordmark">SimplyReview</h1>
        <p className="tagline">Ask questions across your research papers. Every answer draws on both passage search and a knowledge graph.</p>
        {stats && (
          <dl className="stats">
            <div><dt>Papers</dt><dd>{stats.papers}</dd></div>
            <div><dt>Passages</dt><dd>{stats.chunks}</dd></div>
            <div><dt>Entities</dt><dd>{stats.entities}</dd></div>
            <div><dt>Relations</dt><dd>{stats.relations}</dd></div>
          </dl>
        )}
      </header>

      <Library papers={papers} locked={locked} onChanged={refresh} />

      <main className="workspace">
        {loadError && <p className="notice notice-error">{loadError}</p>}

        <form
          className="ask"
          onSubmit={(e) => {
            e.preventDefault();
            ask(question);
          }}
        >
          <label htmlFor="question" className="ask-label">
            Ask your library
          </label>
          <div className="ask-row">
            <textarea
              id="question"
              rows={2}
              value={question}
              placeholder={hasPapers ? "What problem does each paper solve, and how do their approaches differ?" : "Add a paper first, then ask about it here."}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(question);
                }
              }}
            />
            <button type="submit" className="button button-primary" disabled={running || question.trim().length < 3}>
              {running ? "Answering…" : "Ask"}
            </button>
          </div>
          {state.status === "idle" && hasPapers && (
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className="suggestion"
                  onClick={() => {
                    setQuestion(s);
                    ask(s);
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </form>

        {state.status !== "idle" && (
          <>
            <Trace state={state} />
            {state.error && <p className="notice notice-error">{state.error}</p>}
            {state.answer && (
              <Answer text={state.answer} streaming={running} sources={state.sources} onCite={setActiveSource} />
            )}
            <Sources sources={state.sources} hits={state.hits} active={activeSource} />
          </>
        )}
      </main>

      <GraphPanel library={library} answer={state.subgraph} />
    </div>
  );
}
