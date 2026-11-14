"""Okapi BM25 over chunk index texts. Small corpus, so a plain in-memory implementation."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from app.search.text import tokenize


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(c.values()) for c in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.docs) else 0.0
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def term_idf(self, term: str) -> float:
        # Unknown terms get the maximum idf: they are rare by definition.
        return self.idf.get(term, math.log(1 + (len(self.docs) + 0.5) / 0.5))

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.docs))
        if not q:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avgdl or 1.0))
        for term in set(q):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out
