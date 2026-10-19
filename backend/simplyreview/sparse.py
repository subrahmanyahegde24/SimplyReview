"""Pure-Python BM25 sparse encoder for Qdrant keyword search.

Documents carry the BM25 term-frequency component; Qdrant applies the IDF
component server-side (collection created with ``Modifier.IDF``), so the
fused score is a true BM25 score without shipping a vocabulary around.
"""

import hashlib
import re
from collections import Counter

K1 = 1.2
B = 0.75
AVG_DOC_LEN = 180.0  # typical token count of a chunk after stopword removal

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just me more most my no nor
    not now of off on once only or other our ours out over own same she should so some such than that the
    their theirs them then there these they this those through to too under until up very was we were what
    when where which while who whom why will with would you your yours we us via using used use based
    paper show shows shown propose proposed approach results result method methods""".split()
)


def _stem(token: str) -> str:
    # Light plural folding only: keeps model names like "bert" or "resnet50" intact.
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def token_index(token: str) -> int:
    """Stable 31-bit hash so indices match across processes and deployments."""
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "little") & 0x7FFFFFFF


def _to_sparse(weights: dict[int, float]) -> tuple[list[int], list[float]]:
    indices = sorted(weights)
    return indices, [weights[i] for i in indices]


def encode_document(text: str) -> tuple[list[int], list[float]]:
    tokens = tokenize(text)
    if not tokens:
        return [], []
    norm = K1 * (1 - B + B * len(tokens) / AVG_DOC_LEN)
    weights: dict[int, float] = {}
    for token, tf in Counter(tokens).items():
        idx = token_index(token)
        weights[idx] = weights.get(idx, 0.0) + tf * (K1 + 1) / (tf + norm)
    return _to_sparse(weights)


def encode_query(text: str) -> tuple[list[int], list[float]]:
    return _to_sparse({token_index(t): 1.0 for t in set(tokenize(text))})
