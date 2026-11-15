"""Conservative spelling correction for questions ("vacaton days?" -> "vacation days?").

A word is only corrected when it is neither a handbook word nor a common English word, and a
handbook word is within edit distance 1 (2 for words of 7+ letters). Capitalised words inside
the question are left alone: they are usually names ("London"), and an unknown name is a signal
for the answerer to abstain, not something to correct.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

_WORD = re.compile(r"[A-Za-z]+")
_MIN_LEN = 4
# Informal spellings that are fine as they are (and must not become "what" -> "whats").
_INFORMAL = {"whats", "hows", "whos", "wheres", "whens", "dont", "doesnt", "cant", "wont", "isnt",
             "arent", "didnt", "shouldnt", "couldnt", "wouldnt", "thats", "theres", "okay", "thanks", "hello"}


@lru_cache(maxsize=1)
def english_words() -> frozenset[str]:
    path = Path(__file__).with_name("english_words.txt")
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                     if line and not line.startswith("#"))


def edit_distance(a: str, b: str, limit: int) -> int:
    """Optimal string alignment distance (insert, delete, substitute, swap neighbours),
    returning limit + 1 as soon as the distance must exceed `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class SpellingCorrector:
    def __init__(self, texts: list[str], extra_words: tuple[str, ...] = (), user_words: set[str] = frozenset()):
        """texts: handbook text (the correction targets). extra_words: titles and the like.
        user_words: words the search understands although the handbook doesn't use them."""
        counts: Counter = Counter()
        for t in list(texts) + list(extra_words):
            counts.update(w for w in _WORD.findall(t.lower()) if len(w) >= _MIN_LEN)
        for w in user_words:
            counts[w] += 1
        self.vocab = counts
        self.by_len: dict[int, list[tuple[str, int]]] = {}
        for w, f in counts.items():
            self.by_len.setdefault(len(w), []).append((w, f))
        self.known = english_words() | set(counts) | _INFORMAL

    def suggest(self, word: str) -> str | None:
        limit = 1 if len(word) < 7 else 2
        best, best_key = None, None
        for n in range(len(word) - limit, len(word) + limit + 1):
            for cand, freq in self.by_len.get(n, ()):
                d = edit_distance(word, cand, limit)
                if d <= limit and (best_key is None or (d, -freq) < best_key):
                    best, best_key = cand, (d, -freq)
        return best

    def correct(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """Return the corrected text and the (original, replacement) pairs."""
        shouting = text.isupper()
        changes: list[tuple[str, str]] = []
        out, last = [], 0
        for i, m in enumerate(_WORD.finditer(text)):
            w = m.group()
            if len(w) < _MIN_LEN:
                continue
            if not shouting and (w.isupper() or (w[0].isupper() and i > 0)):
                continue                      # acronym or a name: never "correct" it
            lw = w.lower()
            if lw in self.known:
                continue
            fix = self.suggest(lw)
            if fix is None:
                continue
            if w[0].isupper():
                fix = fix.upper() if w.isupper() else fix.capitalize()
            out.append(text[last:m.start()] + fix)
            last = m.end()
            changes.append((w, fix))
        out.append(text[last:])
        return "".join(out), changes
