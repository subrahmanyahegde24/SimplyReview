"""LLM extraction of paper metadata and the entity/relation knowledge graph."""

import re
from typing import Literal

from pydantic import BaseModel, Field

from . import llm
from .documents import Chunk

EntityType = Literal["Method", "Model", "Dataset", "Metric", "Task", "Concept", "Tool"]
RelationType = Literal[
    "PROPOSES",
    "USES",
    "EXTENDS",
    "IMPROVES_ON",
    "OUTPERFORMS",
    "EVALUATED_ON",
    "MEASURED_BY",
    "ADDRESSES",
    "PART_OF",
    "COMPARED_WITH",
    "RELATED_TO",
]

CHUNKS_PER_CALL = 4


class PaperMeta(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    summary: str = Field(description="Two-sentence summary of the paper's contribution")


class Entity(BaseModel):
    name: str = Field(description="Canonical short name, e.g. 'BERT', 'ImageNet', 'F1 score'")
    type: EntityType
    description: str = Field(description="One sentence on what it is, as described in the text")


class Relation(BaseModel):
    source: str = Field(description="Entity name exactly as listed in entities")
    target: str = Field(description="Entity name exactly as listed in entities")
    type: RelationType
    description: str = Field(description="One sentence stating the relationship, with numbers if given")


class ChunkGraph(BaseModel):
    chunk: int = Field(description="The [chunk N] number the facts came from")
    entities: list[Entity]
    relations: list[Relation]


class BatchGraph(BaseModel):
    chunks: list[ChunkGraph]


META_PROMPT = """Extract bibliographic metadata from the opening pages of this research paper.

{text}"""

GRAPH_PROMPT = """You build a knowledge graph of research literature from the paper "{title}".

For each chunk below, list the important technical entities (methods, models, datasets, metrics, tasks,
key concepts, tools) and the relations between them that the chunk states explicitly.

Rules:
- Use the canonical short name ("ResNet-50", not "the ResNet-50 architecture we use").
- The paper's own contribution counts as a Method or Model; name it as the paper does.
- Only relations the text supports. Put quantitative results in the description ("F1 of 91.2 on SQuAD").
- Skip generic words (model, data, experiment), citations, authors, and section headings.
- A chunk with nothing technical gets empty lists.

{chunks}"""


def entity_key(name: str) -> str:
    """Normalised identity used to merge the same entity across chunks and papers."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


async def extract_meta(pages: list[str]) -> PaperMeta:
    text = "\n".join(pages[:2])[:8000]
    return await llm.extract(META_PROMPT.format(text=text), PaperMeta)


async def extract_graph(title: str, chunks: list[Chunk]) -> list[tuple[Chunk, ChunkGraph]]:
    body = "\n\n".join(f"[chunk {c.index}]\n{c.text}" for c in chunks)
    result = await llm.extract(GRAPH_PROMPT.format(title=title, chunks=body), BatchGraph)
    by_index = {c.index: c for c in chunks}
    return [(by_index[g.chunk], g) for g in result.chunks if g.chunk in by_index]
