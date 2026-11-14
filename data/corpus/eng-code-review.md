---
doc_id: eng-code-review
title: Code Review Guidelines
category: Engineering
owner: CTO Office
effective_date: 2024-10-01
status: current
---

# Code Review Guidelines

Every change to a production repository needs **2 approving review(s)** from someone other than the author before merge, plus a green CI pipeline. Reviews should be returned within one working day; if you cannot review in time, say so and suggest someone else.

Review for correctness, clarity and tests first, style last. Prefer small changes under 400 lines. Authors reply to every comment. Approve with comments when the remaining points are minor. Changes to payment flows or robot safety code additionally need a review from the owning team's lead.
