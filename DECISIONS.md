# Decision log

## 1. Approach in five lines

1. One parser per format keeps the author's structure (headings, bullets, table rows, Word body order) in a canonical text with a character offset for every block.
2. A **lifecycle** (in force / superseded / deprecated / not yet effective) is computed per document for the question's date, from the manifest checked against each file's own header.
3. Search fuses BM25 (plus a small thesaurus and typo correction) with `bge-small-en-v1.5` vectors in Qdrant, then re-ranks so stale versions sit below current ones.
4. An LLM (OpenAI `gpt-5.5`; Claude also supported) restates the question in handbook vocabulary, then answers in JSON. Every quote is located in the source or dropped. Without a key, an extractive answerer quotes the best sentences or abstains.
5. FastAPI serves `/api/ask` and `/api/documents/{id}` and a chat page. Citations open the passage highlighted, on the original PDF page for PDFs. Uploads are indexed immediately.

## 2. Key decisions

| Decision | What else I considered | Why I chose this |
|---|---|---|
| **Structure-aware chunks, no overlap:** one chunk per table row or bullet, with its heading path and table header. | Fixed 512-token windows with overlap | A row or bullet is one fact. Fixed windows split the hotel table from its heading and made highlights vague. |
| **Lifecycle computed, not trusted;** stale chunks down-ranked and labelled, and the current sibling always fetched. | Trust the manifest's `status`; delete old versions | The manifest calls v1 and the 2027 policy "current". Deleting old versions breaks "what did the old runbook say?". |
| **Hybrid BM25 + `bge-small` in Qdrant, RRF dense weight 2.0; cross-encoder available but off.** | BM25 or dense alone; model2vec; MiniLM / bge re-rankers | Each choice is an ablation row in `eval/RESULTS.md`. Re-rankers gain little once fusion is tuned, at +1.3–12 s per search on CPU. |
| **Verify quotes myself** (normalised, then ≥0.85 fuzzy match) and replace them with the real source text. | Provider citation features; trusting the model | I need offsets into my own text for highlighting. A fabricated quote can never reach the user. |
| **LLM search planning + whole short documents.** | Bigger embedder; more passages; HyDE | The first LLM run missed "complain about my manager" (Grievance) and city tiers split across sections. |
| **Extractive mode that abstains, and the LLM path can never fail with a 500:** any provider error falls back with a warning. | Require an LLM key | Reviewers may call the backend without a key, or while the API is down. A grounded answer or an honest "not found" beats a guess. |
| **Defensive API for unknown callers:** `query`/`q` aliases, a GET form, typo correction, greetings, upload checks (binary, duplicate, corrupt). | Strict schema only | Reviewers ask their own questions. Every case in the README edge-case table was probed and is tested. |
| **Three graders:** regexes on the key fact, retrieval metrics (hit, MRR, recall, nDCG, P@5), and an LLM judge (faithfulness, relevance, correctness). | Regex only; judge only | The regex is free and deterministic. The judge catches regex false negatives and incomplete answers. Judge disagreements are read by hand, because the judge is the answerer's model family. |
| **Original-PDF viewer** (bundled PDF.js) with character-exact highlights. | Extracted text only; CDN | Reviewers see the real document. A CDN worker is cross-origin and fails offline. |

## 3. Things in the data or the problem that surprised me

- **Leave v1 (12 days) and v2 (16 days) are both "current"** in the manifest; only v2's header and `supersedes` say otherwise.
- **The deprecated 2023 deploy runbook contradicts the current one on four values**, and being shorter, it *wins* plain BM25. Lifecycle re-ranking takes stale-above-current from 50% to 0%.
- **The 2027 parental-leave revision is "current" but future-dated,** and it is an amendment ("all other terms continue"), not a replacement. Answers give today's 12 weeks plus the 26-week change; `as_of` after 2027-01-01 applies it.
- **Referenced but missing:** the Holiday Calendar CSV and the Anti-Harassment Policy. Questions about them say so instead of guessing.
- **Format traps:** python-docx separates tables from their section, and pypdf drops paragraph breaks and wraps header lines. I walk the Word XML in order and rebuild PDF structure. The handbook index also links to `.md` files that are really `.pdf` and `.docx`.
- **Conflicting rules:** the approval matrix says claims over $2,500 need a VP, but travel booked in VoyageHub is exempt.
- **Vocabulary gaps:**
  - "probation" is the "introductory period";
  - "RTO" means Recovery Time Objective, not return to office;
  - Austin and Denver appear in no city tier, so they fall into Tier 3.
- **Name collisions:** three Tylers and three Mayas.

## 4. What I tried that did not work

- **"Semantic rescue"** (accept a sentence at embedding similarity ≥0.40) raised paraphrase accuracy from 25% to 42%. But it answered the unanswerable "book a Denver meeting room" from an irrelevant sentence, so I removed it. This used the held-out set, which is therefore no longer blind.
- **The first extractive answerer scored 75.4%:** lead-in sentences quoted without their tables. Lead-in absorption, anaphora merging and thesaurus-aware coverage took it to 95.7%. Later, a stronger embedder exposed two picker bugs ("resign" matched any "notice"); both are fixed.
- **The first LLM run was too cautious.** It abstained when the answer was future-dated or split across sections. Search planning, whole short documents and a prompt rule took held-out from 92% to 100%. Held-out informed this, so it is not blind.
- **Regex grading flattered the extractive answerer.** The judge rated 85.5% of its dev answers fully correct, against 95.7% by regex. They name the right number but drop a condition (half bonus for interns, manager *and* department head). The judge also found one regex false negative ("6-month" vs `6 months`); I fixed it and regraded without new calls.

## 5. Results

There was no provided dev set; this is my dev set (69 questions). Command: `python -m eval.run_eval --mode llm|extractive --judge`, run as of 2026-11-14. Every metric, set, ablation row and failure is in [`eval/RESULTS.md`](eval/RESULTS.md).

| dev set (69) | LLM mode (`gpt-5.5`) | extractive (no key) |
|---|---|---|
| Answer accuracy, regex grader | **100%** (98.6% before the regex fix) | 95.7% |
| LLM judge: fully correct / correct or partial | 98.6% / 100% | 85.5% / 95.7% |
| Judge faithfulness (claims supported by cited text) | 99.3% | 99.1% |
| Quotes verified verbatim / stale source cited first | 100% / 0% | 100% / 0% |
| Abstention accuracy (7 unanswerable) | 100% | 85.7% |
| Retrieval hit@1 / recall@5 / nDCG@5 / MRR | 98.4 / 99.2 / 98.4 / 99.2 | 96.8 / 99.2 / 97.9 / 98.4 |
| Precision@5 (chunks) | 67.4% | 66.1% |
| Latency p50 / p95 | 4.9 s / 11.6 s | 95 ms / 370 ms |

- **Other sets:**
  - Held-out (25): 100% in both modes (judge: 96% / 92%).
  - Paraphrase (12): LLM 100%, extractive 33.3%.
- **Model and calls:** OpenAI `gpt-5.5` with `reasoning_effort` medium, 2 calls per question. The full run made 212 answer calls and 200 judge calls (judge also `gpt-5.5`).
- **Gaps:**
  - The extractive paraphrase judge did not run because the API credits ran out.
  - Claude is tested with fake clients only; my Anthropic key was rejected.
  - LLM latency varies with the API: held-out p50 was 3.1 s in one run and 10.5 s in this one.

## 6. What I would do with one more week

1. **A judge from a different model family, plus 30 hand-labelled answers.** Today the judge is the answerer's model, checked only against regexes. A labelled sample would measure the judge itself.
2. **Real employee questions and a larger blind set.** All 106 questions are mine, one question moves a metric by 1–8 points, and the held-out set has informed two fixes.
3. **Production storage and access.** Move BM25 into Qdrant sparse vectors so nothing rebuilds at startup, add document deletion and version history, and add per-category access control. Today's in-memory rebuild suits 60 documents, not 60,000.

## 7. Time spent

Done over two days (2026-11-14 and 15) with Claude Code doing the implementation (see AI_USAGE.md). Wall-clock hours come from the session log and include waiting for eval runs.

| Phase | Hours |
|---|---|
| First complete build: corpus read, data problems, ingestion, lifecycle, search, answering, web app, eval, tests, docs | 0.9 |
| Browser testing; LLM mode and its first eval; reviewing the query flow and chunking | 1.0 |
| Qdrant in Docker; comparing the build with my ChatGPT research | 0.3 |
| Improvements: `bge-small`, fusion tuning, re-ranker ablation, PDF.js viewer, precision@5; bugs they exposed; final eval | 2.8 |
| Recall@k, nDCG, LLM judge; edge-case audit and fixes; regrading; this one-page rewrite | 1.6 |
| Repository and docs | 0.3 |
| **Total** | **≈ 6.9** |

ChatGPT research was done alongside and not timed separately.
