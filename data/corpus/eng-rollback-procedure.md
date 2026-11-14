---
doc_id: eng-rollback-procedure
title: Rollback Procedure
category: Engineering
owner: Platform Team
effective_date: 2025-08-01
status: current
---

# Rollback Procedure

Use this when the one-click rollback window in the deploy console has passed.

1. Identify the last known good version tag from the deploy history.
2. Create a rollback deploy from that tag; CI is skipped for tags that were previously deployed.
3. Follow the canary stage exactly as for a normal deploy.
4. Announce in #deploys and link the incident if one is open.
5. Open a follow-up ticket to fix forward; a rollback is never the end of the story.
