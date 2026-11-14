# AI usage

## Tools I used

* **Claude Code (Claude Opus 5.5) in VS Code**, as a coding agent. It read the corpus, wrote all of the code, ran the tests and the eval, and iterated on the results. I did not hand-edit the code; every change came from my instructions and review.
* **ChatGPT**, for research. I used it to learn how production RAG systems are usually built: the pipeline stages, normalising parsed documents into one representation, chunking with metadata, which vector database to use, and which retrieval and re-ranking techniques are common. I gave the resulting plan to Claude Code and used it as a checklist to review the build.

## What I used them for

| Task | Tool | How much of the result I kept |
|---|---|---|
| Reading all 64 documents and cataloguing contradictions, stale versions, missing references and parsing traps | Claude Code | All findings kept (DECISIONS.md section 3); each one checked against the source files |
| Ingestion parsers for the six formats, lifecycle logic, chunking | Claude Code | Kept after fixes driven by inspecting parsed output (Word table order, PDF list numbers, wrapped PDF headers) |
| BM25, dense retrieval, RRF fusion, lifecycle re-rank | Claude Code | Kept. The dense weight was set from a measured sweep, not a default |
| Extractive answerer | Claude Code | Kept after several eval-driven iterations; one idea ("semantic rescue") was removed, see below |
| LLM answerer (OpenAI and Claude), search planning, JSON schema, quote verification | Claude Code | Kept. Measured with OpenAI gpt-5.5; the first LLM run's failures led to search planning and whole-document passages |
| Eval questions (dev, held-out, paraphrase) and harness | Claude Code | Kept. The held-out set was written before any tuning |
| FastAPI backend, chat UI, tests, README, this log | Claude Code | Kept |
| Research: production RAG architecture, vector databases, retrieval and re-ranking techniques | ChatGPT | Used as a review checklist. Four items were applied: `bge-small` embeddings, a measured cross-encoder re-ranker, `.env.example` plus precision@5, and the original-PDF viewer. Not adopted: fixed-size 600-token chunks with 100-token overlap (structure-aware chunks instead, see DECISIONS.md); pgvector (Qdrant instead, see DECISIONS.md); a React frontend (a vanilla JS page needs no build step for one chat screen) |
| Qdrant vector database in Docker | Claude Code, at my request | Kept |

"Kept" means kept after my review and the measured eval. Where I wanted something changed, I asked Claude Code to change it rather than editing the code myself.

## One thing the AI got wrong

**The semantic rescue.** To lift the extractive answerer's weak score on paraphrased questions (25%), the AI added a rule: accept a sentence if its embedding similarity to the question is at least 0.40, even with little word overlap. On the dev and paraphrase sets it looked like a free win: paraphrase accuracy went to 42% with no dev regression.

The final held-out run showed the cost. The unanswerable question "How do I book a meeting room in the Denver office?" was now answered, confidently and with a citation, using an irrelevant Office Locations sentence. Held-out accuracy dropped from 96% to 92%.

We confirmed the cause by re-running that question with the rule disabled: it abstained, with best word coverage 0.36. We then removed the rule entirely. For a handbook assistant, a wrong answer with a citation is worse than an honest "not found".

**A smaller one:** the first text-extraction script read Word files as "all paragraphs, then all tables". That moved the hotel-rate table from section 4.2 to the end of the Travel Policy, so a "Tier 1 = $275" row would have been chunked without its heading. It was caught by reading the extracted text and fixed by walking the document body in order. A test now guards it (`test_docx_table_stays_in_its_section`).

## What I did without AI

I did not write code by hand. My part was direction, research, testing and decisions:

* **Architecture direction.** I researched with ChatGPT how RAG systems are built in industry, then compared Claude Code's build against that plan. The plan's main requirements are in the final design, checked by me against the code:
  - Clean and normalise every format into one standard representation.
  - Chunk by document structure and attach metadata to every chunk.
  - Store embeddings in a vector database.
  - Retrieve with hybrid search plus re-ranking.
  - Let the answer abstain when the evidence is insufficient.
* **Vector database.** I tested the app in the browser, was not satisfied with the results, and asked for a real vector database instead of in-memory vectors. That became Qdrant in Docker. It re-embeds only new or changed chunks and falls back to memory if Docker is down.
* **Questioning the design.** I asked for the full query flow to be explained in plain and technical terms. I also challenged why the build did not use the common "fixed-size chunks with overlap" pattern. I accepted structure-aware chunks only after the trade-off was explained: table rows and list items keep their headers, and highlights stay exact.
* **Choosing improvements.** From the comparison with my research I chose which improvements to build first: better embeddings, a re-ranker experiment, settings plus precision@5, and the PDF viewer. I also accepted the measured result that the re-ranker should stay off by default.
* **LLM mode.** I decided to run the assistant with a real LLM and supplied the API keys. The Anthropic key was rejected, so the measured LLM results use OpenAI `gpt-5.5`.
* **Submission.** I decided what goes into the repository (the code, corpus, eval sets and results, but not the brief and templates) and reviewed these documents.

**What I did not do myself:** the data problems in DECISIONS.md section 3 were found by Claude Code in its first pass, before my research. That includes the stale and superseded versions handled by the lifecycle logic. So were the eval questions and the bug fixes in section 4. I reviewed them but did not originate them.
