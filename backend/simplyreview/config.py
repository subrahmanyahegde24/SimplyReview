"""Runtime configuration, read once from the environment (and backend/.env locally)."""

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    chat_model: str
    extract_model: str
    embed_model: str
    embed_dim: int
    qdrant_url: str
    qdrant_api_key: str | None
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str | None
    admin_token: str | None
    collection_prefix: str
    on_vercel: bool

    @property
    def chunks_collection(self) -> str:
        return f"{self.collection_prefix}chunks"

    @property
    def entities_collection(self) -> str:
        return f"{self.collection_prefix}entities"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


@cache
def get_settings() -> Settings:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    missing = [n for n in ("GEMINI_API_KEY", "QDRANT_URL", "NEO4J_URI", "NEO4J_PASSWORD") if not _env(n)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return Settings(
        gemini_api_key=_env("GEMINI_API_KEY"),
        chat_model=_env("GEMINI_CHAT_MODEL", "gemini-3.5-flash"),
        extract_model=_env("GEMINI_EXTRACT_MODEL", "gemini-3.5-flash-lite"),
        embed_model=_env("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        embed_dim=int(_env("EMBED_DIM", "768")),
        qdrant_url=_env("QDRANT_URL"),
        qdrant_api_key=_env("QDRANT_API_KEY"),
        neo4j_uri=_env("NEO4J_URI"),
        neo4j_username=_env("NEO4J_USERNAME", "neo4j"),
        neo4j_password=_env("NEO4J_PASSWORD"),
        neo4j_database=_env("NEO4J_DATABASE"),
        admin_token=_env("ADMIN_TOKEN"),
        collection_prefix=_env("COLLECTION_PREFIX", "sr_"),
        on_vercel=bool(_env("VERCEL")),
    )
