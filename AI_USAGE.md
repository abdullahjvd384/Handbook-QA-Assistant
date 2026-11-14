# AI usage

## Tools I used

Claude Code (Claude Opus 5.5) in VS Code, as a coding agent that read the corpus, wrote code, ran the tests and the eval, and iterated on the results.

<!-- TODO (candidate): add any other tools you used (ChatGPT, Copilot, ...). -->

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

<!-- TODO (candidate): adjust the "kept" column to reflect what you changed yourself. -->

## One thing the AI got wrong

**The semantic rescue.** To lift the extractive answerer's weak score on paraphrased questions (25%), the AI added a rule: accept a sentence if its embedding similarity to the question is at least 0.40, even with little word overlap. On the dev and paraphrase sets it looked like a free win: paraphrase accuracy went to 42% with no dev regression.

The final held-out run showed the cost. The unanswerable question "How do I book a meeting room in the Denver office?" was now answered, confidently and with a citation, using an irrelevant Office Locations sentence. Held-out accuracy dropped from 96% to 92%.

We confirmed the cause by re-running that question with the rule disabled: it abstained, with best word coverage 0.36. We then removed the rule entirely. For a handbook assistant, a wrong answer with a citation is worse than an honest "not found".

**A smaller one:** the first text-extraction script read Word files as "all paragraphs, then all tables". That moved the hotel-rate table from section 4.2 to the end of the Travel Policy, so a "Tier 1 = $275" row would have been chunked without its heading. It was caught by reading the extracted text and fixed by walking the document body in order. A test now guards it (`test_docx_table_stays_in_its_section`).

## What I did without AI

<!-- TODO (candidate): describe honestly the parts you wrote, decided or verified yourself, e.g.
reviewing the eval questions against the documents, choosing to remove the semantic rescue,
checking the UI in a browser, running Claude mode with your own key. -->
