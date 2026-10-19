"use client";

import { useRef, useState } from "react";
import { streamEvents, type IngestEvent, type Paper } from "@/lib/api";

const TOKEN_KEY = "simplyreview.adminToken";

const STEP_LABEL = {
  parse: "Reading the PDF",
  embed: "Indexing passages",
  extract: "Extracting entities and relations",
  link: "Linking entities",
} as const;

interface Upload {
  name: string;
  step: string;
  done?: number;
  total?: number;
  entities?: number;
  result?: string;
  error?: string;
}

function readToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

function saveToken(token: string) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {}
}

interface Props {
  papers: Paper[];
  locked: boolean;
  onChanged: () => void;
}

export function Library({ papers, locked, onChanged }: Props) {
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [dragging, setDragging] = useState(false);
  const [token, setToken] = useState(readToken);
  const inputRef = useRef<HTMLInputElement>(null);
  const busy = uploads.some((u) => !u.result && !u.error);

  const update = (name: string, patch: Partial<Upload>) =>
    setUploads((list) => list.map((u) => (u.name === name ? { ...u, ...patch } : u)));

  async function ingest(files: File[]) {
    const pdfs = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    setUploads(pdfs.map((f) => ({ name: f.name, step: "Waiting" })));
    for (const file of pdfs) {
      const body = new FormData();
      body.append("file", file);
      try {
        await streamEvents<IngestEvent>(
          "/api/ingest",
          { body, headers: token ? { "x-admin-token": token } : {} },
          (event) => {
            switch (event.type) {
              case "stage":
                if (event.status === "running") update(file.name, { step: STEP_LABEL[event.stage] });
                break;
              case "progress":
                update(file.name, { done: event.done, total: event.total, entities: event.entities });
                break;
              case "meta":
                update(file.name, { step: `Indexing “${event.title}”` });
                break;
              case "done":
                update(file.name, {
                  result: `Added in ${event.seconds} s: ${event.chunks} passages, ${event.entities} entities, ${event.relations} relations`,
                });
                onChanged();
                break;
              case "skipped":
                update(file.name, { result: event.reason });
                break;
              case "error":
                update(file.name, { error: event.message });
                break;
            }
          },
        );
      } catch (err) {
        update(file.name, { error: err instanceof Error ? err.message : String(err) });
      }
    }
  }

  return (
    <aside className="library" aria-label="Library">
      <h2 className="section-title">Library</h2>

      <div
        className="dropzone"
        data-dragging={dragging}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!busy) ingest(Array.from(e.dataTransfer.files));
        }}
      >
        <p>Drop research papers here as PDFs, up to 4.4 MB each.</p>
        <button type="button" className="button" disabled={busy} onClick={() => inputRef.current?.click()}>
          {busy ? "Adding papers…" : "Add papers"}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          multiple
          hidden
          onChange={(e) => {
            ingest(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
        {locked && (
          <label className="token-field">
            Upload key
            <input
              type="password"
              value={token}
              placeholder="Required to add papers"
              onChange={(e) => {
                setToken(e.target.value);
                saveToken(e.target.value);
              }}
            />
          </label>
        )}
      </div>

      {uploads.length > 0 && (
        <ul className="uploads" aria-live="polite">
          {uploads.map((u) => (
            <li key={u.name} data-state={u.error ? "error" : u.result ? "done" : "running"}>
              <span className="upload-name">{u.name}</span>
              <span className="upload-step">{u.error ?? u.result ?? u.step}</span>
              {!u.result && !u.error && u.total ? (
                <span className="upload-progress">
                  <progress value={u.done} max={u.total} />
                  <span>{u.entities} entities so far</span>
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      )}

      {papers.length === 0 ? (
        <p className="empty">No papers yet. Add a few on one topic to start asking questions across them.</p>
      ) : (
        <ul className="papers">
          {papers.map((p) => (
            <li key={p.id}>
              <p className="paper-title">{p.title}</p>
              <p className="paper-meta">
                {[p.authors?.length ? `${p.authors[0]}${p.authors.length > 1 ? " et al." : ""}` : null, p.year]
                  .filter(Boolean)
                  .join(", ")}
              </p>
              <p className="paper-counts">
                {p.pages} pages, {p.entities} entities
              </p>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
