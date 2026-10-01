"""From-scratch NLP primitives: sentence splitting, tokenization,
TF vectors, cosine similarity, and a TextRank-lite extractive summarizer
(sentence similarity graph + PageRank). Standard library only.
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself me more most my myself no nor
    not of off on once only or other ought our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where
    which while who whom why with would you your yours yourself yourselves
    """.split()
)

# A sentence boundary is a period/exclamation/question mark followed by
# whitespace and a capital letter, digit, or quote. Good enough for
# meeting transcripts and fully deterministic.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(sentence: str) -> list[str]:
    """Lowercase word tokens with stopwords removed."""
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", sentence.lower())
    return [w for w in words if w not in STOPWORDS]


def tf_vector(tokens: list[str]) -> Counter:
    """Term-frequency vector (raw counts)."""
    return Counter(tokens)


def cosine_similarity(a: Counter, b: Counter) -> float:
    """Cosine similarity between two sparse TF vectors."""
    if not a or not b:
        return 0.0
    # iterate over the smaller vector
    if len(a) > len(b):
        a, b = b, a
    dot = sum(count * b.get(term, 0) for term, count in a.items())
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(c * c for c in a.values()))
    norm_b = math.sqrt(sum(c * c for c in b.values()))
    return dot / (norm_a * norm_b)


def similarity_matrix(sentences: list[str]) -> list[list[float]]:
    """Pairwise cosine similarity between sentence TF vectors."""
    vectors = [tf_vector(tokenize(s)) for s in sentences]
    n = len(sentences)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            mat[i][j] = sim
            mat[j][i] = sim
    return mat


def pagerank(
    sim: list[list[float]],
    damping: float = 0.85,
    min_diff: float = 1e-5,
    max_iter: int = 200,
    threshold: float = 0.05,
) -> list[float]:
    """PageRank over the sentence-similarity graph.

    Edges exist where similarity exceeds `threshold`; edge weights are the
    similarities. Returns one score per sentence.
    """
    n = len(sim)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    # Outgoing weight per node (TextRank normalizes by the source's total).
    out_weight = [sum(row[j] for j in range(n) if sim[i][j] > threshold) for i, row in enumerate(sim)]
    scores = [1.0 / n] * n
    for _ in range(max_iter):
        new_scores = [(1 - damping) / n] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if sim[j][i] > threshold and out_weight[j] > 0:
                    new_scores[i] += damping * scores[j] * sim[j][i] / out_weight[j]
        diff = sum(abs(a - b) for a, b in zip(scores, new_scores))
        scores = new_scores
        if diff < min_diff:
            break
    return scores


def textrank(
    sentences: list[str],
    top_n: int = 5,
    damping: float = 0.85,
    threshold: float = 0.05,
) -> list[int]:
    """Return indices of the `top_n` ranked sentences, in original order."""
    if not sentences:
        return []
    sim = similarity_matrix(sentences)
    scores = pagerank(sim, damping=damping, threshold=threshold)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top = sorted(ranked[: max(1, min(top_n, len(sentences)))])
    return top


def summarize_sentences(sentences: list[str], top_n: int = 5) -> list[str]:
    """Extractive summary: the `top_n` TextRank sentences in document order."""
    return [sentences[i] for i in textrank(sentences, top_n=top_n)]
