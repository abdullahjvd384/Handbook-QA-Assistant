# Decision log

## 1. Approach in five lines

1. One parser per format keeps the author's structure (headings, bullets, table rows, Word body order) and builds a canonical text with a character offset for every block.
2. Manifest metadata is merged with each file's own header and checked against it. A **lifecycle** (in force / superseded / deprecated / not yet effective) is then computed for an "as of" date.
3. Search runs BM25 (plus a small thesaurus) and `bge-small-en-v1.5` embeddings stored in Qdrant (Docker), fuses them with RRF (dense weight 2.0), then re-ranks so stale versions sit below the current one. A cross-encoder re-ranker is available but off by default.
4. With an API key, an LLM (OpenAI `gpt-5.5` or Claude `claude-opus-5-5`) first restates the question in handbook vocabulary, then answers in JSON-schema output from the passages. Every quote is located in the source; unverifiable quotes are dropped. Without a key, an extractive answerer quotes the best sentences and abstains when unsure.
5. FastAPI serves `/api/ask` and `/api/documents/{id}`, plus a vanilla JS chat page. Citations open the document with the exact span highlighted, on the original PDF page for PDFs (bundled PDF.js) and in the structured text otherwise. Uploads run the same pipeline and swap in a new index snapshot.

## 2. Key decisions

| Decision | What else I considered | Why I chose this |
|---|---|---|
| **Structure-aware chunks.** Sections up to about 120 words. Every table row and bullet is its own chunk, indexed with document title, heading path and table header or list lead-in. | Fixed 512-token windows with overlap; whole documents | Documents are short and dense with facts, and a row or bullet is one fact. Fixed windows separated the Word hotel table from its heading and made highlights vague. |
| **Lifecycle computed, not trusted.** It is derived from `supersedes`, `status` and effective date versus "as of". Stale chunks are down-ranked, labelled in prompts and citations, and their current sibling is always pulled in. | Trust the manifest's `status`; delete old versions at ingest | The manifest calls v1 and the 2027 policy "current". Deleting old versions would break "what did the old runbook say?" and hide the scheduled 2027 change. The answer depends on the date. |
| **Embeddings: `BAAI/bge-small-en-v1.5`** via fastembed (ONNX on CPU, no torch, ~70 MB), at RRF dense weight **2.0**. | `model2vec potion-base-8M` (the first choice: 30 MB, 1 ms); larger bge models | Measured in section 5. bge-small alone reaches 91.7% hit@1 on paraphrases (model2vec: 66.7%). The fusion weight was re-tuned on dev and paraphrase only: 0.5 suited the weak model, but with bge it let BM25 outvote good semantic matches. Cost: ~15–40 ms per search instead of ~1 ms. |
| **Cross-encoder re-ranker available but off** (`HANDBOOK_RERANK=1`; default model MiniLM-L6). | `bge-reranker-base` on by default | With the re-tuned fusion it adds nothing net: dev hit@1 96.8 → 95.2%, paraphrase 83.3 → 91.7% (one question each way), extractive accuracy equal or lower, and +1.5 s per search on CPU (bge-reranker-base: +6–9 s, ~1 GB). MiniLM matched bge-reranker-base at a quarter of the latency, so it is the option if a deployment wants it. |
| **Verify quotes myself.** The model returns a passage id plus a verbatim quote. I locate it (normalised match, else a ≥0.85 fuzzy sentence match) and replace it with the real source text. | Anthropic's citations feature; trusting the model | The citations feature can't be combined with structured output, and I need offsets into *my* canonical text to highlight. This way a fabricated quote can never reach the user. |
| **Extractive fallback that abstains.** | Require an LLM; return raw top chunks | Reviewers may call the backend without a key. A grounded answer or an honest "not found" beats an LLM-shaped guess. |
| **Regex-graded eval** with gold and stale documents, in dev / held-out / paraphrase sets. | LLM-as-judge; exact match | Deterministic, free and repeatable. The regexes check the specific fact, and the held-out set (written before tuning) checks I didn't fit to my own questions. |
| **Stateless follow-ups.** The client sends `history`. The LLM's planning call resolves it into a standalone question; extractive mode uses a heuristic. | Server-side sessions | Simpler, easy for external callers, scales horizontally. |
| **LLM search planning plus whole short documents.** One low-effort call restates the question in handbook vocabulary, given the list of document titles; both phrasings are searched and fused. A short document (≤700 words) with two or more hits is passed whole. | A bigger embedding model; more passages; HyDE | The first LLM run (section 4) failed on "complain about my manager" (Grievance Policy not in the top 30) and on city-tier lookups split across sections. The title list grounds the rewrite in documents that actually exist. Cost: one extra small call per question. |
| **Qdrant for vectors, kept in sync by content hash.** Each point stores a hash of model plus chunk text; startup and uploads embed only new or changed chunks and delete stale ones. In-memory fallback when Qdrant is down. BM25 stays in memory. | Exact NumPy search only; Chroma / FAISS / pgvector; Qdrant sparse vectors for BM25 too | Persistence across restarts and a path to scale and to several app instances. Qdrant runs as one Docker container with a dashboard. The fallback keeps the app usable without Docker. On all 106 eval questions Qdrant returned the same top 8 as exact in-memory search, so quality is unchanged; this is a storage and operations choice, not an accuracy one. |
| **Original-PDF viewer with character-exact highlighting.** PDF.js 3.11 renders the real page; the quote is matched on the text layer (whitespace and quote-mark insensitive) and only the quoted characters are wrapped, even mid-line. Bundled under `static/vendor`. | Show only extracted text; PDF page plus whole-line highlight; PDF.js from a CDN | Reviewers see the document as written, not my parse of it. Whole-span highlighting overshot into the next sentence. A CDN worker is cross-origin, which browsers refuse for Web Workers, and it breaks offline. Checked in headless Chrome on 4 PDFs and the full ask → click-citation flow. |
| **Provider-independent LLM layer** (OpenAI and Anthropic, same prompt, schema and verification). | One SDK only | The working key available was OpenAI. Verification doesn't depend on the provider, so swapping providers can't weaken grounding. |

## 3. Things in the data or the problem that surprised me

- **Leave v1 vs v2:** 12 vs 16 days of annual leave. Both are "current" in the manifest; only v2's PDF header and the manifest's `supersedes` field say otherwise. I compute supersession and flag the conflict in the ingest report. Answers say 16 days.
- **The deprecated 2023 deploy runbook contradicts the current one on four values:** freeze 36h vs 72h, 1 vs 2 reviews, no canary, rollback 10 vs 45 min. Being shorter, it *wins* plain BM25 for "rollback window" (there's a test for this). Lifecycle re-ranking takes "stale doc ranked above current" from 58% to 0%.
- **The 2027 parental leave revision is marked "current" but takes effect in the future.** Answers give 12 weeks today and mention 26 weeks for births from 2027-01-01. The eval pins `as_of = 2026-11-14`.
- **Referenced documents that don't exist:** the Holiday Calendar CSV and the Anti-Harassment Policy are detected. Questions about them get "not in the handbook", and Claude is told they're missing.
- **Broken handbook index:** it links to `hr-leave-policy-v2.md` and `hr-travel-policy.md`, but the real files are `.pdf` and `.docx`. Reported.
- **Word table out of order:** python-docx's `paragraphs` and `tables` lists separate the hotel-cap table from section 4.2. I walk the body XML in order instead.
- **PDF text extraction loses structure:** pypdf drops blank lines, puts list numbers on their own lines, and wraps "Supersedes:" onto the next line. I rebuild paragraphs, lists and the header.
- **Expense policy vs approval matrix:** the matrix says claims over $2,500 need a VP; the expense policy exempts travel booked in VoyageHub. The specific exception wins, and there's a test question for it.
- **Terminology:**
  - "RTO" means Recovery Time Objective (the glossary itself warns it's not return-to-office).
  - "Probation" is called the "introductory period".
  - Austin and Denver appear in no city tier, so they're Tier 3 via "any city not listed".
- **Name collisions:** three Tylers, three Mayas and two Riveras. Org-chart rows are separate chunks so full names match exactly.
- **This machine has Python 3.15,** which has no wheels for parts of the stack. The project is pinned to 3.11–3.13.

## 4. What I tried that did not work

- **First extractive answerer, 75.4% on dev.** It abstained on "How much vacation do I get?". It quoted "capped per night by city tier:" without the table, and "After that window, rollback requires…" without the sentence it refers to.
  - Fixes:
    - thesaurus-aware coverage;
    - a lead-in sentence absorbs the rows that follow;
    - anaphoric sentences are merged with the one before;
    - question framing ("how long") is stripped;
    - answer-type words are down-weighted;
    - newer policy revisions are always fetched.
  - Result: dev **95.7%**.
- **Unknown-term abstention fired too often.** Abstaining when a question word appears nowhere in the handbook also fired on "wear" in "What do I need to wear in the lab?". I restricted it to capitalised names and acronyms (CEO, Wi-Fi, London).
- **Equal-weight fusion hurt.** Dense-only retrieval had the worst hit@1 (75.8%), and fusing at weight 1.0 lowered dev hit@1 (93.5% → 88.7%). A weight sweep chose 0.5.
- **"Semantic rescue" was removed.** It let the extractive answerer accept a sentence whose embedding similarity was at least 0.40 despite low word overlap.
  - Gain: paraphrase accuracy 25% → 42%.
  - Cost: it answered the unanswerable held-out question "book a meeting room in Denver" with an irrelevant Office Locations quote, dropping held-out accuracy from 96% to 92%.
  - I removed it: a confident, cited, wrong answer is worse than abstaining. *Caveat:* this one decision used the held-out set, so that set is no longer fully blind on this point.
- **The stronger embedder exposed two bugs in the extractive picker.** Switching to bge-small first *dropped* extractive dev accuracy (95.7 → 92.8%), although retrieval improved.
  - Bug 1: "How much vacation…" was answered with the carry-forward sentence. Both sentences fully covered "annual leave days", so passage order decided. Fix: a small bonus when the unit's section heading matches the question ("2. Annual leave").
  - Bug 2: "How much notice if I resign?" was answered with the parental-leave notice rule. The thesaurus maps "resign" to "resignation notice", and the coverage check counted the word "notice" alone as a match. Fix: a mapped phrase must actually appear (both words of a two-word phrase).
  - A tried variant, using the document title as the heading for documents without sections, broke the rollback question and was reverted.
  - Result: 95.7% on dev with every embedder and weight tested, and held-out 96 → 100% (the dress-code false positive was the same synonym bug).
- **The first LLM run was too cautious and saw too little.** Dev 97.1%, held-out 92.0%, paraphrase 83.3%.
  - It abstained on Austin and Seattle hotel caps: the Travel Policy's "City tiers" section ranked 11th or lower, behind its other fragments.
  - It marked "can leave be split?" unanswerable because today's policy is silent, even though the 2027 revision allows it.
  - It missed the Grievance and Performance Review documents for casual wording.
  - Fixes: search planning, whole short documents, and a prompt rule that partial or future-dated answers are answerable. Result: 98.6% / 100% / 100%. *Caveat:* these held-out failures informed the fixes, so held-out is no longer blind for the LLM numbers.

## 5. Results

There was no provided dev set; these are my three sets (`eval/*.jsonl`). Run: `python -m eval.run_eval --mode llm|extractive [--ablation] --questions …`, as of 2026-11-14. Results are in `eval/results/<set>/results-<mode>.json`. Final configuration: bge-small at dense weight 2.0, Qdrant, no re-ranker.

**LLM mode: OpenAI `gpt-5.5`, Chat Completions, strict JSON schema, `reasoning_effort` medium (low for search planning).**

| metric | dev (69) | held-out (25)* | paraphrase (12) |
|---|---|---|---|
| answer accuracy | **98.6%**† | **100%** | **100%** |
| abstention accuracy (unanswerable) | 100% | 100% | – |
| citation support / precision | 100% / 97.8% | 100% / 100% | 100% / 93.5% |
| quotes verified verbatim | 100% | 100% | 100% |
| stale doc as primary citation | 0% | 0% | 0% |
| retrieval hit@1 / hit@3 / MRR / P@5 | 93.5 / 100 / 96.8 / 66.8 | 87.0 / 95.7 / 91.7 / 44.3 | 83.3 / 83.3 / 86.8 / 45.0 |
| latency p50 / p95 | 3.5 / 6.5 s | 3.1 / 5.3 s | 4.8 / 11.1 s |
| model calls | 138 | 50 | 24 |

\* Not blind for the LLM numbers; see section 4.
† The one miss is a grader false negative. The answer said "a **Security team assessment**", and the regex expected `security assessment`. I widened the regex afterwards; the saved results file still shows the strict grade.

**Extractive mode (no key, 0 model calls):**

| metric | dev (69) | held-out (25) | paraphrase (12) |
|---|---|---|---|
| answer accuracy | 95.7% | 100% | 33.3% |
| abstention accuracy | 85.7% | 100% | – |
| quotes verified / stale primary | 100% / 0% | 100% / 0% | 100% / – |
| retrieval hit@1 / MRR / P@5 | 96.8 / 98.4 / 66.1 | 91.3 / 93.5 / 45.2 | 83.3 / 83.3 / 41.7 |
| latency p50 | 57 ms | 64 ms | 69 ms |

Extractive mode still fails where reasoning is needed: date arithmetic (the next freeze), an inference across two documents (per diem needs no claim), noticing a passage doesn't answer (the dental *provider*), and most casual wording. The LLM fixes these.

**Retrieval ablation.** Dev set, with paraphrase hit@1 in brackets. Extractive answers, 0 LLM calls, search latency on CPU:

| configuration | hit@1 | MRR | P@5 | stale ranked above current | answer acc. | search ms |
|---|---|---|---|---|---|---|
| BM25 | 82.3 (50.0) | 89.8 | 61.0 | 50.0% | 94.2% | 1 |
| BM25 + thesaurus | 83.9 (50.0) | 91.4 | 61.3 | 58.3% | 95.7% | 1 |
| BM25 + thesaurus + lifecycle | 93.5 (50.0) | 96.5 | 63.9 | **0%** | 95.7% | 2 |
| dense only, model2vec | 75.8 (66.7) | 86.8 | 52.9 | 33.3% | 94.2% | 3 |
| hybrid + lifecycle, model2vec, w=0.5 *(previous default)* | 91.9 (58.3) | 95.7 | 62.6 | 0% | 95.7% | 3 |
| dense only, bge-small | 83.9 (**91.7**) | 91.7 | 59.7 | 66.7% | 94.2% | 41 |
| hybrid RRF, bge-small, w=2.0 | 85.5 (75.0) | 92.5 | 62.3 | 58.3% | 95.7% | 68 |
| **hybrid + lifecycle, bge-small, w=2.0 (default)** | **96.8** (83.3) | **98.4** | **66.1** | **0%** | 95.7% | 28 |
| … + re-rank MiniLM-L6 | 95.2 (91.7) | 97.6 | 60.6 | 0% | 94.2% | 1,578 |
| … + re-rank bge-reranker-base | 95.2 (91.7) | 97.6 | 60.6 | 0% | 95.7% | 5,758 |

What the ablation shows:

- **Lifecycle re-ranking** is the largest single gain on this corpus: stale-above-current goes from 50–67% to 0%, and hit@1 rises by about 10 points.
- **bge-small** is better than model2vec on every set, most of all on paraphrases.
- **Fusion weight:** dense weight 2.0 versus 0.5 with bge, tuned on dev and paraphrase only. Dev hit@1 93.5 → 96.8%, paraphrase 58.3 → 83.3%.
- **Cross-encoders** didn't pay for their latency once fusion was tuned. Held-out (not used for tuning) agrees: hit@1 is 91.3% without re-ranking and 100% with it, but answer accuracy is 100% either way.
- **Qdrant** returned the same top 8 as exact in-memory search for all 106 questions.

**Model, provider and calls:**

- The LLM numbers use **OpenAI `gpt-5.5`**: 2 calls per question (search planning plus answer), **212 calls** for all three sets.
- The Anthropic path (`claude-opus-5-5`) uses the same prompt and schema but is **not measured**: the Anthropic key I had was rejected (401). It is covered by fake-client tests.
- Extractive mode makes 0 calls.

## 6. What I would do with one more week

1. **An LLM judge for faithfulness, and a cost/latency pass.** Regexes check the key fact but not whether every sentence of an answer is supported. Two calls per question at 3–5 s is fine for chat but could be cut: skip planning when the keyword match is strong, or use a smaller model for planning.
2. **Get real questions and a bigger blind set.** Every question here was written by me after reading the corpus, and with 106 questions one question moves a metric 1–8 points. The held-out set has now informed decisions twice (section 4), so it isn't blind any more. Mining real chat questions would test the vocabulary gap honestly.
3. **Production storage and access.** Vectors already persist in Qdrant. Next: keep BM25 in Qdrant too (sparse vectors) so nothing is rebuilt at startup; add document deletion and version history in the UI; add per-category access control as payload filters. Today the BM25 index and document texts are rebuilt in memory at startup, which is fine for 60 documents, not 60,000.

## 7. Time spent

The work was done in one day (2026-11-14) with Claude Code doing the implementation (see AI_USAGE.md). The hours are wall-clock time from the session log. They include waiting for eval and ablation runs. Because an agent did the coding, these are not comparable to hand-written hours, and they don't split cleanly by component, so they're grouped by phase.

| Phase | Hours |
|---|---|
| First complete build: brief and corpus read, data problems catalogued, ingestion and lifecycle, hybrid search, extractive answers, web app, eval sets, tests, docs | 0.9 |
| Testing the app in the browser; LLM mode (OpenAI) and its first eval; reviewing the query flow and chunking choices | 1.0 |
| Qdrant in Docker; comparing the build with my ChatGPT research | 0.3 |
| Improvements from that comparison: `bge-small` embeddings, re-tuned fusion, re-ranker ablation, `.env.example` and precision@5, PDF.js viewer; extractive bugs it exposed; final eval on all sets; docs | 2.8 |
| Preparing the repository and filling in these logs | 0.3 |
| **Total** | **≈ 5.3** |

ChatGPT research was done alongside the session and not timed separately.
