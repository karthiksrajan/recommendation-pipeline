"""Parsing of the messy Goodreads `genres` field (29-genre boilerplate prefix, repeated twice)."""
from __future__ import annotations

import pandas as pd

NON_CATEGORY = {"Fiction", "Nonfiction", "Audiobook", "Ebooks", "Audiobooks", "Adult", "Novels", "Contemporary",
                "Book Club", "Adult Fiction", "Childrens", "Literature", "American", "School", "Academic"}


class GenreParser:
    """Learns the boilerplate prefix from the data, then returns the de-duplicated list of *real* genres."""

    def __init__(self, prefix):
        self.prefix = list(prefix)

    @classmethod
    def fit(cls, genres: pd.Series, n_prefix: int = 29) -> "GenreParser":
        return cls(genres.dropna().iloc[0].split(",")[:n_prefix])

    def prefix_share(self, genres: pd.Series) -> float:
        n = len(self.prefix)
        return float(genres.dropna().map(lambda s: s.split(",")[:n] == self.prefix).mean())

    def parse(self, s):
        if not isinstance(s, str) or not s.strip():
            return []
        toks = [t.strip() for t in s.split(",") if t.strip()]
        n = len(self.prefix)
        for _ in range(2):                                   # strip the repeated boilerplate prefix
            if toks[:n] == self.prefix:
                toks = toks[n:]
        seen, out = set(), []
        for t in toks:
            if "Goodreads" in t:                             # scraping artefact such as "Goodreads' Transgender Books page"
                continue
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @staticmethod
    def primary(genres):
        for g in genres:
            if g not in NON_CATEGORY:
                return g
        return None
