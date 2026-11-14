"""Tokenisation shared by BM25, sentence scoring and the eval."""
from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

_stemmer = snowballstemmer.stemmer("english")

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his
how i if in into is it its itself just me more most my myself no nor not of off on once only or other our ours
out over own same she should so some such than that the their theirs them then there these they this those
through to too under until up very was we were what when where which while who whom why will with would you
your yours get got many much need needs tell know please ok okay hi hello also now currently im ive use using
""".split())
NUMBER_WORDS = frozenset("one two three four five six seven eight nine ten eleven twelve".split())
# Agent nouns the Snowball stemmer leaves alone ("owner" vs "owns").
_LEMMAS = {"owner": "own", "owners": "own", "approver": "approve", "approvers": "approve",
           "reviewer": "review", "reviewers": "review"}

# Words that carry meaning in a handbook even though they look like stopwords.
KEEP = frozenset({"not", "no", "before", "after", "above", "under", "over", "only"})

_NUM_COMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"[a-z0-9]+")
# Spelling variants of the same compound, normalised on both the index and the query side.
_COMPOUNDS = [(re.compile(p), r) for p, r in [
    (r"\broll(?:ed|ing|s)?[\s-]+back\b", "rollback"),
    (r"\bon[\s-]call\b", "oncall"),
    (r"\bsign(?:ed)?[\s-]off\b", "signoff"),
    (r"\bper[\s-]diems?\b", "perdiem"),
    (r"\bwi[\s-]?fi\b", "wifi"),
    (r"\bstand[\s-]up\b", "standup"),
    (r"\be[\s-]?mail\b", "email"),
]]
# "How long ...", "how quickly ..." describe the *kind* of answer wanted, not its topic.
_QUESTION_FRAME = re.compile(r"\bhow (long|quickly|fast|soon|often|far|much|many)\b|\bwhat time\b", re.I)
# Words naming the type of answer; the answer text rarely contains them ("address" -> "1700 Platte Street").
ANSWER_TYPE_WORDS = frozenset({
    "address", "name", "number", "date", "time", "email", "phone", "extension", "amount", "cost", "price",
    "limit", "length", "duration", "deadline", "minimum", "maximum", "max", "min", "rate", "size",
    "percentage", "role", "title", "policy", "rule", "rules", "process", "procedure", "say", "says",
})


@lru_cache(maxsize=50_000)
def stem(word: str) -> str:
    if word.isdigit() or len(word) <= 2:
        return word
    return _stemmer.stemWord(_LEMMAS.get(word, word))


ANSWER_TYPE_STEMS = frozenset(stem(w) for w in ANSWER_TYPE_WORDS)


def normalize(text: str) -> str:
    text = _NUM_COMMA.sub("", text.lower())
    for pat, rep in _COMPOUNDS:
        text = pat.sub(rep, text)
    return text


def strip_question_frame(query: str) -> str:
    return _QUESTION_FRAME.sub(" ", query)


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out = []
    for tok in _TOKEN.findall(normalize(text)):
        if len(tok) == 1 and not tok.isdigit():
            continue
        if not keep_stopwords and tok in STOPWORDS and tok not in KEEP:
            continue
        out.append(stem(tok))
    return out


# Small hand-written thesaurus for words employees use that the handbook doesn't.
# Applied to the keyword (BM25) query only; the dense retriever handles general paraphrase.
# Its effect is measured separately in the eval ablation (SearchConfig.expand_query).
_EXPANSIONS: list[tuple[re.Pattern, str]] = [(re.compile(p, re.I), add) for p, add in [
    (r"\b(vacation|pto|paid time off|time off|days off)\b", "annual leave"),
    (r"\bprobation(ary)?\b", "introductory period"),
    (r"\bmaternity\b", "birth parent leave"),
    (r"\b(paternity|fathers?|dads?|partner)\b", "secondary caregiver leave"),
    (r"\b(wfh|work(ing)? from home|home office)\b", "remote work home office"),
    (r"\b(quit|quitting|resign|resigning|leave the company|leaving the company)\b", "resignation notice"),
    (r"\b(mean|meaning|stand for|stands for|definition|define|acronym)\b", "glossary"),
    (r"\b(daily allowance|meal allowance|perdiem)\b", "per diem"),
    (r"\b(refund|paid back|pay me back)\b", "reimbursement"),
    (r"\b(abroad|overseas|internationally|another country|other country|foreign country|outside the (us|u\.s\.|united states)"
     r"|canada|mexico|uk|united kingdom|england|ireland|europe|germany|france|spain|italy|netherlands|portugal|poland"
     r"|india|pakistan|china|japan|singapore|philippines|australia|brazil|argentina)\b", "outside united states"),
    (r"\b(boss|line manager)\b", "direct manager"),
    (r"\b(computer|notebook|macbook)\b", "laptop"),
    (r"\b(outage|incident) (level|severity)\b", "sev severity"),
]]


def expansions(query: str) -> list[tuple[str, str]]:
    """[(words the user typed, handbook words they map to), ...]"""
    out = []
    for pat, add in _EXPANSIONS:
        m = pat.search(query)
        if m:
            out.append((m.group(0), add))
    return out


def expand_query(query: str) -> str:
    extra = [add for _, add in expansions(query)]
    return f"{query} {' '.join(extra)}" if extra else query


def content_terms(text: str) -> list[str]:
    """Query terms that should be matched by an answer (no stopwords, no 'not'/'only' fillers,
    no question framing such as 'how long')."""
    return [t for t in tokenize(strip_question_frame(text)) if t not in _KEEP_STEMS]


_CORE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def core_question(text: str) -> str:
    """For multi-sentence input ('I'm on a SEV1. How do I reach X?' or an injected instruction
    followed by a question) return the sentence that is actually the question."""
    parts = [p.strip() for p in _CORE_SPLIT.split(text.strip()) if p.strip()]
    asks = [p for p in parts if p.endswith("?")]
    if len(parts) > 1 and asks and len(content_terms(asks[-1])) >= 2:
        return asks[-1]
    return text


def capitalized_terms(text: str) -> set[str]:
    """Stems of capitalised words / acronyms that are not sentence-initial ('CEO', 'London', 'Wi-Fi')."""
    out = set()
    for sent in _CORE_SPLIT.split(text):
        words = re.findall(r"[A-Za-z][\w-]*", sent)
        for w in words[1:]:
            if w[0].isupper() and w.lower() not in STOPWORDS:
                out.update(tokenize(w))
    return out


_KEEP_STEMS = frozenset(stem(w) for w in KEEP)
