"""Bulk-ingest PDFs from disk: ``uv run python ingest_cli.py papers/*.pdf``.

Use this for large files or many papers; it runs the same pipeline as the dashboard upload.
"""

import asyncio
import sys
from pathlib import Path

from simplyreview.ingest import ingest_pdf


async def main(paths: list[Path]) -> None:
    for path in paths:
        print(f"\n== {path.name}")
        async for event in ingest_pdf(path.read_bytes(), path.name):
            match event["type"]:
                case "progress":
                    print(f"   extract {event['done']}/{event['total']}  entities={event['entities']}", end="\r")
                case "stage" if event["status"] == "done":
                    print(f"   {event['stage']:<8} done {' '.join(f'{k}={v}' for k, v in event.items() if k not in ('type', 'stage', 'status'))}")
                case "meta":
                    print(f"   title: {event['title']} ({event.get('year')})")
                case "done" | "skipped" | "error":
                    print(f"   {event}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: uv run python ingest_cli.py <file.pdf> [...]")
    asyncio.run(main([Path(p) for p in sys.argv[1:]]))
