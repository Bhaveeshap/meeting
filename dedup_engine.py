"""
LoopKeeper — Module 2: Semantic De-duplication & State Matching Engine.

Given a newly-extracted action item, decide whether it is:
  (a) the SAME underlying task as an existing active item (possibly
      re-phrased, e.g. "Draft project pitch" vs "Finish slide deck for
      pitch"), in which case it should UPDATE that item, or
  (b) a genuinely NEW task, in which case a new action_item should be
      created.

Matching combines two signals:
  1. Semantic similarity of the action description (embeddings + cosine
     similarity).
  2. Fuzzy match of the assignee name (handles "Bhaveesha" vs "Bhaveesha K."
     vs a typo), used both as a gate (don't match across clearly different
     people) and as a score component.

Embedding backend is pluggable (`EmbeddingBackend` protocol) so this module
works offline out of the box and can be upgraded to real sentence embeddings
in production without touching the matching logic below.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from models import ActionItem, StateStore

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Cosine similarity above which two descriptions are considered a semantic match.
# NOTE: this threshold is calibrated for the default HashingVectorizer backend,
# which is LEXICAL (word-overlap based), not truly semantic. It catches
# paraphrases that share vocabulary ("pitch deck" / "slide deck for the
# pitch") but will miss paraphrases with zero shared words. Swap in
# `SentenceTransformerBackend` (below) and re-tune this threshold for
# production-grade semantic matching (e.g. sentence-transformers cosine
# similarities for genuine paraphrases typically land at 0.6-0.9).
SEMANTIC_MATCH_THRESHOLD = 0.35

# Assignee name fuzzy score (0-100) below which we refuse to match at all,
# even if the text is semantically identical — prevents "draft the report"
# from Alice silently merging into Bob's near-identical task.
ASSIGNEE_FUZZY_GATE = 65.0

# If no match is found for the same assignee, we optionally widen the search
# across ALL active items (to catch reassignments, "hey can you take this
# over from Alice"), but require much higher semantic similarity to do so.
CROSS_ASSIGNEE_SEMANTIC_THRESHOLD = 0.55

_STOPWORDS = {
    "the", "a", "an", "to", "for", "on", "of", "and", "with", "in", "by",
    "please", "should", "will", "need", "needs", "needed",
}


def normalize_action_core(description: str) -> str:
    """
    Cheap normalized form used as a fast pre-filter (pg_trgm in Postgres)
    before the more expensive vector search. Lowercase, strip punctuation,
    drop common stopwords — keeps content words that matter for matching.
    """
    text = description.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    return " ".join(tokens)


def fuzzy_name_score(a: str, b: str) -> float:
    """0-100 similarity between two names, robust to minor variants/typos."""
    a_n, b_n = a.strip().lower(), b.strip().lower()
    if a_n == b_n:
        return 100.0
    # Reward one name being a prefix/substring of the other, e.g.
    # "Bhaveesha" vs "Bhaveesha K." — SequenceMatcher alone underrates this.
    if a_n in b_n or b_n in a_n:
        return 92.0
    return SequenceMatcher(None, a_n, b_n).ratio() * 100.0


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class EmbeddingBackend(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dimension) float array of unit-normalized vectors."""
        ...


class HashingTfidfBackend:
    """
    Default, dependency-light backend: scikit-learn's HashingVectorizer.
    Deterministic, needs no model download/network access and no `fit()`
    step (unlike TfidfVectorizer), which matters for a system that must
    embed items one meeting at a time as they stream in.

    Trade-off: this is a LEXICAL representation (bag-of-words hashed into a
    fixed-size space), not a learned semantic embedding — it will not catch
    paraphrases with no shared vocabulary (e.g. "get sign-off from finance"
    vs "confirm the budget is approved"). It's a reasonable, honest default
    for a fully offline demo; swap in `SentenceTransformerBackend` below for
    real semantic matching in production.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vectorizer = HashingVectorizer(
            n_features=dimension,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),      # bigrams help catch reordered phrases
            stop_words="english",    # drop "the/for/to/of" noise so overlap reflects
                                      # content words, not shared grammar
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts)
        return np.asarray(matrix.todense(), dtype=np.float32)


class SentenceTransformerBackend:
    """
    Production-grade semantic backend using sentence-transformers
    (e.g. 'all-MiniLM-L6-v2', 384-dim — matches the `vector(384)` column in
    schema.sql). Requires `pip install sentence-transformers` and a one-time
    model download, so it's kept optional/import-guarded here rather than a
    hard dependency of the pipeline.

        pip install sentence-transformers
        engine = DedupEngine(store, backend=SentenceTransformerBackend())

    Re-tune SEMANTIC_MATCH_THRESHOLD upward (~0.6-0.7) when using this
    backend — real sentence embeddings produce much higher cosine
    similarities for genuine paraphrases than the hashing fallback does.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SentenceTransformerBackend requires `pip install sentence-transformers`."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    matched_item: Optional[ActionItem]
    semantic_score: float
    assignee_score: float
    combined_score: float
    is_match: bool
    reason: str


class DedupEngine:
    def __init__(self, store: StateStore, backend: Optional[EmbeddingBackend] = None):
        self.store = store
        self.backend = backend or HashingTfidfBackend()

    def embed(self, description: str) -> np.ndarray:
        return self.backend.embed([description])[0]

    def find_match(self, *, assignee_name: str, description: str,
                    embedding: Optional[np.ndarray] = None,
                    allow_cross_assignee: bool = True) -> MatchResult:
        """
        Core matching call. Tries same-assignee candidates first (the
        common case); if nothing clears the bar there and
        `allow_cross_assignee` is set, widens to all active items with a
        stricter semantic bar (handles reassignment/hand-off phrasing).
        """
        embedding = embedding if embedding is not None else self.embed(description)

        same_owner_assignee = self.store.get_or_create_assignee(assignee_name)
        candidates = self.store.active_items_for_assignee(same_owner_assignee.id)

        best = self._best_candidate(embedding, assignee_name, candidates)
        if best is not None and best.combined_score >= 0 and best.is_match:
            return best

        if allow_cross_assignee:
            other_candidates = [
                ai for ai in self.store.all_active_items()
                if ai.assignee_id != same_owner_assignee.id
            ]
            cross_best = self._best_candidate(
                embedding, assignee_name, other_candidates,
                semantic_threshold=CROSS_ASSIGNEE_SEMANTIC_THRESHOLD,
                assignee_gate=0.0,  # name mismatch is expected here by definition
            )
            if cross_best is not None and cross_best.is_match:
                cross_best.reason = "cross-assignee semantic match (possible reassignment): " + cross_best.reason
                return cross_best

        return MatchResult(None, 0.0, 0.0, 0.0, False, "no candidate cleared match thresholds")

    def _best_candidate(self, embedding: np.ndarray, assignee_name: str,
                         candidates: list[ActionItem],
                         semantic_threshold: float = SEMANTIC_MATCH_THRESHOLD,
                         assignee_gate: float = ASSIGNEE_FUZZY_GATE) -> Optional[MatchResult]:
        best_result: Optional[MatchResult] = None

        for candidate in candidates:
            if candidate.embedding is None:
                continue
            sem_score = cosine_similarity(embedding, candidate.embedding)
            assignee_score = fuzzy_name_score(
                assignee_name, self.store.assignees[candidate.assignee_id].canonical_name
            )

            if assignee_score < assignee_gate:
                continue  # hard gate: different people, don't even consider it

            combined = 0.75 * sem_score + 0.25 * (assignee_score / 100.0)
            is_match = sem_score >= semantic_threshold

            if best_result is None or combined > best_result.combined_score:
                best_result = MatchResult(
                    matched_item=candidate,
                    semantic_score=sem_score,
                    assignee_score=assignee_score,
                    combined_score=combined,
                    is_match=is_match,
                    reason=(f"semantic={sem_score:.2f} (threshold {semantic_threshold}), "
                            f"assignee_fuzzy={assignee_score:.0f}"),
                )

        return best_result
