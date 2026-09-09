"""
The search index itself.

Approach: TF-IDF over each chunk's text, then Truncated SVD (latent semantic
analysis) to fold that sparse lexical space down into a small dense space.
This is a real, classic semantic-search technique, not a lexical keyword
match: two chunks that share few exact words but talk about related concepts
(e.g. "db session" and "SQLAlchemy engine") end up closer together in the SVD
space than raw TF-IDF cosine similarity alone would put them, because SVD
groups terms that tend to co-occur across the corpus.

It deliberately does NOT depend on downloading a pretrained neural embedding
model. That keeps the whole thing runnable offline, with no model weights to
fetch and no GPU required, at the cost of not being as strong as a modern
sentence-transformer embedding on genuinely novel vocabulary. `Chunk`-level
docstrings/comments help close that gap since they add natural-language
context around the code's identifier names.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk, chunk_repository

_IDENTIFIER_SPLIT_RE = re.compile(r"[_\W]+")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize_identifiers(text: str) -> str:
    """
    Expands snake_case and camelCase identifiers into separate words before
    TF-IDF tokenizes them, e.g. `risk_score` and `RiskScore` both become
    "risk score" tokens. Without this, a query for "risk score" would never
    match a chunk that only ever writes `risk_score` as one token.
    """
    out_tokens = []
    for raw in _IDENTIFIER_SPLIT_RE.split(text):
        if not raw:
            continue
        camel_parts = _CAMEL_SPLIT_RE.sub(" ", raw).split()
        out_tokens.extend(p.lower() for p in camel_parts if p)
    return " ".join(out_tokens)


class SearchIndex:
    def __init__(self, n_components: int = 128, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.use_svd: bool = True
        self.doc_vectors = None  # np.ndarray (SVD) or sparse matrix (TF-IDF fallback)
        self.chunks: list[Chunk] = []
        self.build_stats: dict = {}

    # ---- building ----

    def build(self, root: str) -> dict:
        t0 = time.time()
        chunks = chunk_repository(root)
        if not chunks:
            raise ValueError(f"No indexable files found under {root!r}")

        corpus = [_tokenize_identifiers(c.text) for c in chunks]
        n_samples = len(corpus)

        # max_df=0.95 (ignore terms in >95% of docs) is only meaningful once
        # there are enough documents for "95%" to mean something; on a
        # handful of chunks it can mathematically conflict with min_df and
        # sklearn raises. Only apply it once the corpus is big enough.
        max_df = 0.95 if n_samples >= 20 else 1.0

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_df=max_df,
            min_df=1,
            ngram_range=(1, 2),
        )
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # TruncatedSVD needs strictly more samples and features than the
        # number of components it's asked to find, or the decomposition is
        # degenerate (sklearn will happily return NaNs for the explained
        # variance instead of raising). Below that size, semantic dimension
        # reduction doesn't have enough data to mean anything anyway, so we
        # fall back to plain TF-IDF cosine similarity, which is well-defined
        # even for a single-document corpus.
        max_possible_components = min(tfidf_matrix.shape[0], tfidf_matrix.shape[1]) - 1
        n_components = min(self.n_components, max_possible_components)

        if n_components >= 2:
            self.use_svd = True
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            self.doc_vectors = self.svd.fit_transform(tfidf_matrix)
            explained = float(self.svd.explained_variance_ratio_.sum())
        else:
            self.use_svd = False
            self.svd = None
            self.doc_vectors = tfidf_matrix
            explained = 1.0  # raw TF-IDF cosine similarity uses 100% of the lexical signal

        self.chunks = chunks
        elapsed = time.time() - t0

        self.build_stats = {
            "root": root,
            "num_chunks": len(chunks),
            "num_files": len({c.file_path for c in chunks}),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "svd_components": n_components if self.use_svd else 0,
            "used_svd": self.use_svd,
            "explained_variance_ratio": round(explained, 4),
            "build_seconds": round(elapsed, 3),
        }
        return self.build_stats

    # ---- querying ----

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.vectorizer is None or self.doc_vectors is None:
            raise RuntimeError("Index has not been built yet. Call .build(root) first.")

        q_tokens = _tokenize_identifiers(query)
        q_tfidf = self.vectorizer.transform([q_tokens])
        q_vec = self.svd.transform(q_tfidf) if self.use_svd else q_tfidf

        sims = cosine_similarity(q_vec, self.doc_vectors)[0]
        top_idx = np.argsort(-sims)[:top_k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            chunk = self.chunks[idx]
            d = chunk.to_dict()
            d["score"] = round(float(sims[idx]), 4)
            d["rank"] = rank
            # keep results readable: cap very long chunks in the response
            if len(d["text"]) > 1200:
                d["text"] = d["text"][:1200] + "\n... (truncated)"
            results.append(d)
        return results

    # ---- persistence ----

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_components": self.n_components,
                    "random_state": self.random_state,
                    "vectorizer": self.vectorizer,
                    "svd": self.svd,
                    "use_svd": self.use_svd,
                    "doc_vectors": self.doc_vectors,
                    "chunks": [asdict(c) for c in self.chunks],
                    "build_stats": self.build_stats,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "SearchIndex":
        with open(path, "rb") as f:
            state = pickle.load(f)
        idx = cls(n_components=state["n_components"], random_state=state["random_state"])
        idx.vectorizer = state["vectorizer"]
        idx.svd = state["svd"]
        idx.use_svd = state.get("use_svd", True)
        idx.doc_vectors = state["doc_vectors"]
        idx.chunks = [Chunk(**c) for c in state["chunks"]]
        idx.build_stats = state["build_stats"]
        return idx
