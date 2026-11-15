# Northgate Handbook Q&A assistant

Answers employee questions from the Northgate Robotics handbook. Every answer quotes the passage it rests on, and every citation opens the source document with that passage highlighted.

The handbook mixes Markdown, CSV, plain text, HTML, PDF and Word files. Some documents are out of date and some contradict each other, so the assistant works out which version applies on a given date (superseded, deprecated, or not yet in effect) and answers from the version in force.

* **Backend:** FastAPI. `POST /api/ask` answers a question; `GET /api/documents/{doc_id}` fetches a document.
* **Search:** BM25 plus `BAAI/bge-small-en-v1.5` embeddings stored in a **Qdrant** vector database (Docker), fused with Reciprocal Rank Fusion, then a lifecycle-aware re-rank. An optional cross-encoder re-ranker is available.
* **Answers:**
  * With an API key (`OPENAI_API_KEY` for `gpt-5.5`, or `ANTHROPIC_API_KEY` for `claude-opus-5-5`), an LLM first restates the question in handbook vocabulary for searching, then writes the answer with structured output. Every quote is checked against the source text, and unverifiable quotes are dropped.
  * With no key, a deterministic extractive answerer quotes the best-matching sentences and abstains when unsure. The backend always works.
* **Web page:** chat with clickable citations and follow-up questions. The document viewer highlights the cited passage, on the **original PDF page** for PDFs (PDF.js, bundled, works offline) and in the extracted text for every format. Document upload is searchable immediately, with no restart.
* **Evaluation:** 106 hand-written questions in three sets (dev, held-out, paraphrase), a scoring script, and a retrieval ablation.

Design choices and results are in [DECISIONS.md](DECISIONS.md). How AI tools were used is in [AI_USAGE.md](AI_USAGE.md).

---

## 1. Install

Requires **Python 3.11–3.13**. It was developed on 3.12; Python 3.15 lacks wheels for some dependencies.

```bash
cd handbook-qa
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

With [uv](https://docs.astral.sh/uv/): `uv venv --python 3.12 .venv && uv pip install -r requirements.txt`.

**Start the vector database** (Qdrant, in Docker):

```bash
docker compose up -d          # Qdrant on http://localhost:6333 (dashboard: /dashboard); vectors persist in a Docker volume
```

The first app start embeds all chunks into Qdrant. Later starts re-embed only chunks that are new or changed. Without Docker the app still works: it logs a warning and keeps the vectors in memory (results are identical, they just aren't persisted).

On first start the embedding model (`BAAI/bge-small-en-v1.5`, ONNX, about 70 MB) downloads and is cached in `~/.cache/handbook-qa/models` (override with `HANDBOOK_MODEL_CACHE`). If it can't be downloaded (for example offline), search falls back to BM25 only and `/api/health` reports `"dense_search": false`.

**Settings:** copy `.env.example` to `.env` and fill in what you need: API key, models, re-ranker, Qdrant URL. Shell environment variables take precedence. `.env` is git-ignored; never commit keys.

**Optional:** to get LLM-written answers, set **one** key before starting. OpenAI is what the reported results use.

```bash
export OPENAI_API_KEY=sk-proj-...          # Windows PowerShell: $env:OPENAI_API_KEY="sk-proj-..."
# or: export ANTHROPIC_API_KEY=sk-ant-...
```

If both are set, Anthropic is used unless `HANDBOOK_LLM_PROVIDER=openai`. Don't commit keys: the app reads them only from the environment.

## 2. Run the tests

```bash
python -m pytest -q          # 69 tests, about 2-3 min, no API key needed (3 Qdrant tests skip if Docker isn't running)
```

The tests cover:

* parsing of every format, including table order in the Word file and wrapped PDF header lines;
* lifecycle rules: supersession, deprecation, and future effective dates;
* retrieval, including the stale-version re-rank;
* the extractive answerer and quote verification;
* the LLM path (OpenAI and Anthropic), using fake clients, so dropped quotes, fallback, search planning and whole-document passages are all exercised;
* the HTTP API, including upload and immediate querying.

## 3. Run it

```bash
python -m app.ingest -v      # ingest the corpus; print chunks, lifecycle per document and data-quality findings
python -m eval.run_eval      # run the evaluation (section 6)
```

## 4. Start the web app

```bash
uvicorn app.api:app --port 8000
```

Open <http://localhost:8000>. Ask a question, then click a `[n]` marker or a source card to open the document with the cited passage highlighted.

* **PDFs** open on the original page, rendered with PDF.js, with the exact quoted characters highlighted. An **Extracted text** tab shows the parsed text.
* **Other formats** open in the structured text view.
* **Links are shareable:** `/?doc=<id>&start=<offset>&end=<offset>`.

Other buttons:

* **Documents** lists every document with its lifecycle.
* **Upload** sends a new file through the same ingestion pipeline and makes it searchable straight away.

Follow-up questions ("and for secondary caregivers?") use the conversation. **New chat** clears it.

---

## 5. Question endpoint (for external evaluation)

### `POST /api/ask`

Request body (JSON). Only `question` is required (`query` and `q` are accepted as aliases).

| field | type | default | meaning |
|---|---|---|---|
| `question` | string, 1–2000 chars, not blank | (required) | The question. |
| `history` | list of `{role: "user" \| "assistant", content}` | `[]` | Earlier turns, oldest first, for follow-ups. |
| `as_of` | `YYYY-MM-DD` | today | Answer as of this date. This decides which policy versions are in force. |
| `mode` | `"auto"` \| `"llm"` \| `"extractive"` | `"auto"` | `auto` uses the LLM when an API key is set, otherwise extractive. |
| `top_k` | int 1–20 | 8 | Passages retrieved. |

```bash
curl -s http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of annual leave do I get?"}'

# the same as a GET, handy in a browser
curl -s "http://localhost:8000/api/ask?q=How+many+days+of+annual+leave+do+I+get"
```

CORS is open for `GET` and `POST`, so the API can be called from a notebook or another web page.

Response (abridged):

```json
{
  "question": "How many days of annual leave do I get?",
  "standalone_question": "How many days of annual leave do I get?",
  "search_query": null,
  "answer": "According to the Leave Policy v2 (2. Annual leave): “Full-time employees accrue 16 days of annual leave per calendar year, credited each pay period in equal parts. Part-time employees accrue pro rata.” [1]",
  "answerable": true,
  "citations": [
    {
      "n": 1,
      "doc_id": "hr-leave-policy-v2",
      "title": "Leave Policy v2",
      "section": "2. Annual leave",
      "quote": "Full-time employees accrue 16 days of annual leave per calendar year, credited each pay period in equal parts. Part-time employees accrue pro rata.",
      "start": 371, "end": 518,
      "lifecycle": "in_force",
      "effective_date": "2025-04-01",
      "role": "primary",
      "url": "/?doc=hr-leave-policy-v2&start=371&end=518"
    }
  ],
  "mode": "extractive",
  "model": null,
  "as_of": "2026-11-14",
  "spelling_corrections": [],
  "retrieved": [{"doc_id": "hr-leave-policy-v2", "title": "Leave Policy v2", "chunk_id": "hr-leave-policy-v2#2",
                 "section": "2. Annual leave", "score": 0.02407, "lifecycle": "in_force"}],
  "llm_calls": 0,
  "latency_ms": 4,
  "warnings": []
}
```

How to read the response:

* **`answer`** is plain text. `[n]` markers refer to `citations[n-1]`.
* **`citations[].quote`** is always exactly `document.text[start:end]` of the cited document. Fetch it with `GET /api/documents/{doc_id}` to check.
* **`citations[].role`** is one of:
  * `primary`: the main source;
  * `supporting`: an additional source;
  * `upcoming`: a revision that is not in force yet;
  * `outdated`: a superseded or deprecated version, quoted only to flag it.
* **`citations[].lifecycle`** is `in_force`, `superseded`, `deprecated` or `not_yet_effective`, as of `as_of`.
* **`answerable: false`** means the handbook doesn't cover the question. `citations` may then be empty, or point at the sentence that mentions a document missing from the handbook.
* **`mode`** is `llm`, `extractive`, `extractive (fallback)` or `small-talk`. The fallback is used when the LLM call fails or its quotes can't be verified; `warnings` says why. `small-talk` is a greeting or "what can you do?", answered without searching or model calls. `model` names the LLM used.
* **`spelling_corrections`** lists typos fixed before searching, for example `[{"from": "vacaton", "to": "vacation"}]`. `standalone_question` shows the corrected text.
* **`standalone_question`** is the question as searched; for follow-ups it is the rewritten question. In LLM mode, **`search_query`** is the handbook-vocabulary restatement that was searched as well. `llm_calls` is 2 per question in LLM mode.
* The example above is extractive mode. In LLM mode, `answer` is a written answer such as "You get 16 days of annual leave per calendar year as a full-time employee.[1]".

Errors:

* `422`: the request failed validation (for example an empty or blank question, or more than 2000 characters).
* `400`: `mode: "llm"` was requested but no API key is configured.
* The LLM path never returns `500`: any provider error (timeout, rate limit, refusal, invalid JSON, SDK exception) falls back to the extractive answer, with a warning.

Interactive API docs are at <http://localhost:8000/docs>.

### Edge cases and how they are handled

Each row was probed through the running API and is covered by a test (`tests/`).

| input | behaviour |
|---|---|
| Empty or whitespace-only question; over 2000 characters; over 40 history turns | `422` with a validation message. |
| "hi", "thanks", "what can you do?" | A short help reply (`mode: "small-talk"`), no search, no model calls. A greeting followed by a question ("Hi, how many leave days…") is answered normally. |
| Typos: "vacaton", "probaton", "pasword", "ANUAL" | Corrected for search and listed in `spelling_corrections`. Only words that are neither handbook words nor common English words are touched, and never capitalised names or acronyms ("London", "CEO"). |
| Accents and case: "café", ALL CAPS | Folded and lower-cased before matching. |
| Not in the handbook: CEO salary, the weather, a London office | `answerable: false` and a statement of what the handbook does cover. Unknown names make the extractive answerer abstain rather than guess. |
| Gibberish ("asdfghjkl") | `answerable: false`. |
| A document the handbook mentions but doesn't contain (Anti-Harassment Policy) | Says it is referenced but not available, and cites the sentence that mentions it. |
| An old version asked about explicitly ("under the old policy", "the 2023 runbook") | Answers from that version, labelled superseded or deprecated, and gives the current value. |
| A policy that changes in future (2027 parental leave) | Today's rule first, then the change and its date. With `as_of` after the date, the new rule applies. |
| Prompt injection in the question or in `history` ("ignore your rules…") | LLM mode treats passages and conversation as data and answers only from the handbook. Extractive mode cannot follow instructions at all. |
| A question in another language | LLM mode answers in that language from the English handbook. Extractive mode abstains (keyword matching is English-only). |
| Two questions in one ("Who is the Security Lead and what is the password rotation rule?") | LLM mode answers both, each with its own citation. Extractive mode answers the better-covered part. |
| A follow-up without its history ("what about contractors?") | Treated as a standalone question. |
| LLM returns a quote that isn't in the source | The quote is dropped; if no verified citation is left, the extractive answer is used. |
| Qdrant not running / embedding model unavailable | In-memory vectors / BM25 only; `/api/health` says which. |
| Unknown `doc_id`, `../` in a document path | `404`. |
| Upload: wrong type, over 10 MB, empty file | `415`, `413`, `400`. |
| Upload: binary data renamed `.md`, corrupt or scanned (text-less) PDF | `422` with the reason; nothing is indexed. |
| Upload: a file already in the handbook (same bytes) | `409` naming the existing document. |
| Upload: `supersedes` an unknown document; `../` in the file name | `400`; the path is stripped from the name. |
| Upload: HTML with `<script>` | Scripts and styles are dropped at ingest; the page renders all text with `textContent`, never `innerHTML`. |
| Questions while an upload is being indexed | The index is rebuilt off to the side and swapped in atomically; uploads are serialised by a lock. |

### Other endpoints

| method & path | purpose |
|---|---|
| `GET /api/documents/{doc_id}` | Document metadata and lifecycle, the full canonical `text`, and `blocks` (kind, level, page, start, end) used for rendering and highlighting. |
| `GET /api/documents/{doc_id}/file` | The original file (PDF, DOCX and so on). |
| `GET /api/documents?as_of=` | All documents with declared status and computed lifecycle. |
| `POST /api/documents` | Upload, multipart: `file` (required); `title`, `category`, `effective_date`, `supersedes` (optional). Supported: `.md .txt .csv .html .pdf .docx`, up to 10 MB. Returns the new `doc_id`. |
| `GET /api/report` | Data-quality findings from ingestion. |
| `GET /api/health` | Index size, vector store in use (`qdrant` / `memory`) and its sync stats, active answer mode. |

## 6. Evaluation

```bash
python -m eval.run_eval                                    # dev set (69 questions), answer mode auto
python -m eval.run_eval --mode extractive --ablation       # also compare 6 retrieval configurations
python -m eval.run_eval --questions eval/heldout.jsonl     # held-out set (25), written before tuning
python -m eval.run_eval --questions eval/paraphrase.jsonl  # paraphrase probe (12)
python -m eval.run_eval --mode llm                         # LLM answers (needs OPENAI_API_KEY or ANTHROPIC_API_KEY)
python -m eval.run_eval --mode llm --judge                 # plus an LLM judge: faithfulness, relevance, correctness
python -m eval.run_eval --base-url http://localhost:8000   # same questions through the running HTTP API
python -m eval.run_eval --mode llm --regrade               # re-apply edited regexes to saved answers (no model calls)
python -m eval.report                                      # rebuild eval/RESULTS.md from the saved results
```

Each question line in `eval/*.jsonl` holds:

* the question and an expected answer, plus `must_include` / `must_not_include` regexes;
* `supporting_docs` (gold) and `acceptable_docs`;
* `stale_docs`: versions that must not be the primary source;
* optional `history` (follow-ups), `as_of`, and `answerable: false`.

The eval date is pinned to 2026-11-14 so results don't change when the 2027 policy takes effect.

The script prints a summary, per-category and per-format accuracy, and every failure with the expected and actual answer. It writes JSON to `eval/results/<set>/results-<mode>.json`.

The metrics:

| metric | what it checks |
|---|---|
| answer accuracy | All `must_include` regexes match and no `must_not_include` matches. Unanswerable questions must abstain or explain (`accept_if`). |
| citation support / precision | At least one citation is from a gold or acceptable document / the share of citations that are. |
| quote verified | The cited quote is exactly the source text at the given offsets. |
| stale primary | The first citation comes from a superseded or deprecated document (should be 0). |
| hit@k, MRR | A gold document is among the top-k retrieved documents / 1 ÷ rank of the first one. |
| recall@k | The share of all gold documents found in the top k (differs from hit@k on multi-document questions). |
| nDCG@5 | Rank-weighted retrieval quality: gold documents near the top count more. |
| precision@5 | The share of the top 5 retrieved chunks that come from a gold or acceptable document. |
| stale above current | A stale version is ranked above the current one. |
| latency p50 / p95, model calls | Time per question and API calls per run. |
| **LLM judge** (`--judge`) | One extra call per answer. The judge sees the question, the date, the reference answer, the answer and the cited passages with context. |
| judge faithfulness | The answer is split into atomic claims; the share of factual claims the cited text supports (RAGAS-style). |
| judge relevance | Whether the answer addresses the question (full / partial / none). |
| judge correctness | Semantic comparison with the reference answer; catches correct answers that a regex misses ("6-month" vs "6 months"). |
| judge agreement | How often the judge and the regex grader agree. Every disagreement is printed and listed in `eval/RESULTS.md` to be read by hand. |

Latest results. Every metric, the ablation and every failure are in [`eval/RESULTS.md`](eval/RESULTS.md); caveats are in DECISIONS.md section 5.

| | dev (69) | held-out (25) | paraphrase (12) |
|---|---|---|---|
| answer accuracy, LLM mode (`gpt-5.5`, 2 calls per question) | **100%** | **100%** | **100%** |
| answer accuracy, extractive mode (no key) | 95.7% | 100% | 33.3% |
| LLM judge, fully correct: LLM / extractive | 98.6% / 85.5% | 96.0% / 92.0% | 100% / not run |
| judge faithfulness: LLM / extractive | 99.3% / 99.1% | 100% / 99.3% | 99.1% / not run |
| retrieval hit@1 / recall@5 / nDCG@5 (LLM mode) | 98.4 / 99.2 / 98.4 | 87.0 / 100 / 93.7 | 83.3 / 91.7 / 86.9 |
| quote verified (both modes) | 100% | 100% | 100% |
| stale source as primary citation (both modes) | 0% | 0% | 0% / – |

The ablation (`--ablation`) compares 11 retrieval configurations: BM25 / dense / hybrid, two embedding models, and two cross-encoder re-rankers.

## 7. Configuration

| env var | default | |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | (unset) | Either enables LLM answers in `auto` mode. |
| `HANDBOOK_LLM_PROVIDER` | `auto` | `auto` (Anthropic if its key is set, else OpenAI) / `openai` / `anthropic`. |
| `HANDBOOK_ANSWER_MODE` | `auto` | `auto` / `llm` / `extractive`. |
| `HANDBOOK_OPENAI_MODEL` | `gpt-5.5` | OpenAI model ID. |
| `HANDBOOK_LLM_MODEL` | `claude-opus-5-5` | Claude model ID. |
| `HANDBOOK_LLM_EFFORT` | `medium` | `low` / `medium` / `high` (Claude effort / OpenAI `reasoning_effort`). |
| `HANDBOOK_LLM_FALLBACKS` | `1` | Claude only: server-side refusal fallback (beta `server-side-fallback-2026-07-01`). |
| `HANDBOOK_TODAY` | real date | Override "today" for lifecycle decisions. |
| `HANDBOOK_DENSE` | `1` | Set `0` to disable dense search (BM25 only, fully offline). |
| `HANDBOOK_DENSE_MODEL` | `fastembed:BAAI/bge-small-en-v1.5` | Embedding model, `fastembed:<name>` or `model2vec:<name>`. Changing it re-embeds the Qdrant collection automatically. |
| `HANDBOOK_RERANK` / `HANDBOOK_RERANK_MODEL` | `0` / `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder re-ranking of the top 20 candidates. Off by default; see DECISIONS.md. |
| `HANDBOOK_VECTOR_STORE` | `auto` | `auto` (Qdrant if reachable, else memory) / `qdrant` (fail if unavailable) / `memory`. |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://localhost:6333` / `handbook_chunks` | Qdrant location and collection. |
| `HANDBOOK_CORPUS_DIR` / `HANDBOOK_UPLOADS_DIR` | `data/corpus` / `data/uploads` | Where documents are read from and uploads are stored. |

## 8. Layout

```
app/
  api.py              FastAPI app: endpoints + static UI
  config.py           settings from env vars
  models.py           Document, Block, Chunk, Lifecycle, Citation
  ingest/loaders.py   one parser per format -> structural blocks + declared metadata
  ingest/pipeline.py  metadata merge, canonical text with offsets, lifecycle, chunking, quality report
  search/             tokenizer + thesaurus, spelling correction, BM25, embeddings (fastembed / model2vec),
                      re-ranker, vector store (Qdrant / memory), hybrid index
  qa/                 passages, extractive answerer, LLM answerers (OpenAI / Claude), quote verification, service
  static/             chat page (vanilla JS, no build step); vendor/ holds PDF.js 3.11.174 (Apache-2.0)
eval/                 question sets, run_eval.py (harness), judge.py (LLM judge), report.py,
                      results/ (raw JSON) and RESULTS.md (generated tables, every failure)
tests/                pytest suite
data/corpus/          the handbook as delivered (+ manifest.json)
data/uploads/         documents uploaded through the UI (persisted across restarts)
docker-compose.yml    Qdrant vector database
```
