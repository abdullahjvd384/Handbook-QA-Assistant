---
doc_id: eng-feature-flags
title: Feature Flag Process
category: Engineering
owner: Platform Team
effective_date: 2025-01-20
status: current
---

# Feature Flag Process

New user-facing behaviour ships behind a flag, off by default. Flags are turned on progressively: internal users, then 10 percent, then 50 percent, then everyone, with at least one Thursday release between stages for customer-facing changes. Flags older than 90 days are reviewed in the fortnightly platform meeting and removed once fully on or fully off. The flag owner is named in the flag's description.
