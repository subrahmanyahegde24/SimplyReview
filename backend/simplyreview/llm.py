"""Thin async wrapper over the Gemini API: embeddings, structured extraction, streamed answers."""

import asyncio
import random
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import cache
from typing import Literal

from pydantic import BaseModel

from .config import get_settings

EMBED_BATCH = 100  # API maximum per embed request
MAX_RETRIES = 6
MAX_RETRY_WAIT = 65.0

_RETRY_AFTER_RE = re.compile(r"retry(?:Delay': '| in )([\d.]+)s")

EmbedTask = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


@cache
def _client():
    from google import genai

    return genai.Client(api_key=get_settings().gemini_api_key)


def _is_retryable(exc: Exception) -> bool:
    from google.genai import errors

    return isinstance(exc, errors.APIError) and exc.code in (429, 500, 502, 503, 504)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Honour the server's RetryInfo on quota errors (free tier asks for ~30-60s); else back off."""
    if match := _RETRY_AFTER_RE.search(str(exc)):
        return min(float(match.group(1)), MAX_RETRY_WAIT) + random.random()
    return min(2**attempt, 20) + random.random()


async def _with_retry[T](call: Callable[[], Awaitable[T]]) -> T:
    for attempt in range(MAX_RETRIES):
        try:
            return await call()
        except Exception as exc:
            if attempt == MAX_RETRIES - 1 or not _is_retryable(exc):
                raise
            await asyncio.sleep(_retry_delay(exc, attempt))
    raise AssertionError("unreachable")


async def embed(texts: list[str], task: EmbedTask) -> list[list[float]]:
    from google.genai import types

    settings = get_settings()
    config = types.EmbedContentConfig(output_dimensionality=settings.embed_dim, task_type=task)

    async def batch(part: list[str]) -> list[list[float]]:
        response = await _with_retry(
            lambda: _client().aio.models.embed_content(model=settings.embed_model, contents=part, config=config)
        )
        return [e.values for e in response.embeddings]

    batches = [texts[i : i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
    results = await asyncio.gather(*(batch(b) for b in batches))
    return [vector for part in results for vector in part]


async def embed_query(text: str) -> list[float]:
    return (await embed([text], "RETRIEVAL_QUERY"))[0]


async def extract[M: BaseModel](prompt: str, schema: type[M]) -> M:
    """Single structured-output call on the cheap extraction model."""
    from google.genai import types

    settings = get_settings()
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )
    response = await _with_retry(
        lambda: _client().aio.models.generate_content(model=settings.extract_model, contents=prompt, config=config)
    )
    if isinstance(response.parsed, schema):
        return response.parsed
    return schema.model_validate_json(response.text or "{}")


async def stream_answer(system: str, prompt: str) -> AsyncIterator[str]:
    from google.genai import types

    settings = get_settings()
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    stream = await _with_retry(
        lambda: _client().aio.models.generate_content_stream(model=settings.chat_model, contents=prompt, config=config)
    )
    async for chunk in stream:
        if chunk.text:
            yield chunk.text
