---
doc_id: eng-db-migration-runbook
title: Database Migration Runbook
category: Engineering
owner: Data Team
effective_date: 2025-03-10
status: current
---

# Database Migration Runbook

Schema migrations on production databases run in the maintenance window starting at **03:00 Central Time**. Every migration must be reversible or must ship with a documented restore path. Take a snapshot immediately before running the migration and verify that the snapshot is restorable in staging first.

Migrations that lock a table for more than five seconds are split into online steps using the expand and contract pattern. Announce the migration in #deploys the day before. The Data Team on-call monitors replication lag during the migration and aborts if lag exceeds 60 seconds.
