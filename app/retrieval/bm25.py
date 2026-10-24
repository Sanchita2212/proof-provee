import math
import re
from typing import List, Sequence


_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_\-]{2,}\b")


def bm25_scores(query: str, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Return Okapi BM25 scores for each text against a shared query."""
    if not documents:
        return []

    query_terms = _TOKEN_PATTERN.findall(query.lower())
    if not query_terms:
        return [0.0] * len(documents)

    tokenized = [_TOKEN_PATTERN.findall(text.lower()) for text in documents]
    document_frequencies = {}
    for tokens in tokenized:
        for term in set(tokens):
            document_frequencies[term] = document_frequencies.get(term, 0) + 1

    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    scores = []
    for tokens in tokenized:
        term_frequencies = {}
        for term in tokens:
            term_frequencies[term] = term_frequencies.get(term, 0) + 1

        score = 0.0
        length_norm = 1 - b + b * len(tokens) / average_length if average_length else 1.0
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = document_frequencies[term]
            inverse_frequency = math.log(
                1 + (len(tokenized) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            score += inverse_frequency * frequency * (k1 + 1) / (frequency + k1 * length_norm)
        scores.append(score)
    return scores
