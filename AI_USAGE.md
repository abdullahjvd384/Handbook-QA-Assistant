# AI usage

## Tools I used

* **Claude Code (Claude Opus 5.5) in VS Code**, as the coding agent. It wrote all of the code, ran the tests and the evals, and iterated on the results. I did not hand-edit code; every change came from my instructions and review.
* **ChatGPT**, for research: how production RAG systems are built, which vector database to use, and which retrieval, re-ranking and evaluation techniques are standard. I gave that plan to Claude Code and used it as a checklist to review the build.
* The OpenAI API (`gpt-5.5`) is part of the product, as the answer model and the eval's LLM judge. It was not a development aid.

## What I used them for

| Task | Tool | How much of the result I kept, and how it was checked |
|---|---|---|
| Reading the 64 documents and listing contradictions, stale versions, missing references and parsing traps | Claude Code | All kept (DECISIONS.md section 3). Each one checked against the source file. |
| Parsers for six formats, lifecycle logic, structure-aware chunking | Claude Code | Kept after fixes found by reading the parsed output (Word table order, PDF lists, wrapped PDF headers). Unit tests guard each fix. |
| Hybrid search (BM25 + `bge-small` in Qdrant), RRF fusion, lifecycle re-rank, optional cross-encoder | Claude Code | Kept. Every component is justified by an ablation row, not by default. The re-ranker stays off because it measured no better. |
| LLM answering (OpenAI and Claude), search planning, quote verification | Claude Code | Kept. Invented quotes cannot reach the user (verified offsets), and the first LLM run's failures drove search planning. |
| Extractive no-key answerer | Claude Code | Kept after eval-driven iterations. One idea was removed (below). |
| Eval sets (106 questions) and harness | Claude Code | Kept. The held-out set was written before tuning. Regexes were checked against the documents. |
| Production RAG research: architecture, vector DB, retrieval and evaluation metrics | ChatGPT | Used as a review checklist. Applied: `bge-small`, a measured re-ranker experiment, `.env.example`, precision@5, the original-PDF viewer. Not adopted, with reasons: fixed 600-token chunks with overlap, pgvector, React. |
| Edge-case audit and hardening | Claude Code, at my request | Kept. Each case was probed through the real API, then turned into a test (README section 5). |
| Recall@k, nDCG@5 and an LLM-as-judge (faithfulness, relevance, correctness) | Claude Code, at my request | Kept. The judge was checked against planted wrong answers and against the regex grader; every disagreement was read (`eval/RESULTS.md`). |

## One thing the AI got wrong

**The "semantic rescue".** To lift the no-key answerer on paraphrased questions (25%), the AI added a rule: accept a sentence whose embedding similarity to the question is at least 0.40, even with little word overlap. On dev and paraphrase it looked free: paraphrase accuracy went to 42% with no dev regression.

The held-out run showed the cost. The unanswerable "How do I book a meeting room in the Denver office?" was now answered, confidently and with a citation, from an irrelevant Office Locations sentence. Held-out accuracy fell from 96% to 92%. Re-running that question with the rule off confirmed the cause: it abstained. We removed the rule. A cited wrong answer is worse than an honest "not found".

**A smaller one, caught by probing the API:** the new spelling corrector turned "matenity leave" into "paternity leave". It read words out of the thesaurus regexes and kept the `\b` prefix, so "maternity" was never in its vocabulary. Fixed, and a test asserts the right correction.

## What I did without AI

I didn't write code by hand. My part was direction, research, testing and decisions:

* **Architecture direction.** I researched with ChatGPT how RAG systems are built, then reviewed Claude Code's build against that plan. Its requirements are all in the final design:
  - one normalised representation for every format;
  - structure-aware chunks with metadata;
  - a vector database;
  - hybrid retrieval plus re-ranking;
  - abstaining on weak evidence.
* **Vector database.** I tested the app in the browser, wasn't satisfied, and asked for a real vector database instead of in-memory vectors. That became Qdrant in Docker.
* **Questioning the design.** I had the query flow explained to me. I challenged why the build didn't use fixed-size chunks with overlap, and accepted structure-aware chunks once the trade-off was clear.
* **Choosing what to build next.** I chose:
  - which improvements came first: embeddings, a re-ranker experiment, the PDF viewer;
  - that the eval needed recall@k, nDCG and an LLM-as-judge;
  - that reviewer-style edge cases had to be handled and tested.
* **LLM mode.** I decided to run with a real LLM and supplied the keys. The Anthropic key was rejected, so measured results use OpenAI `gpt-5.5`.
* **Submission.** I decided what goes into the repository and reviewed these documents.

**What I did not originate:** the data problems in DECISIONS.md section 3, including the stale and superseded versions handled by the lifecycle logic, were found by Claude Code in its first pass, before my research. So were the eval questions and the bug fixes in section 4. I reviewed them.
