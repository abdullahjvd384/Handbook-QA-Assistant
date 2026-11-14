---
doc_id: eng-deploy-runbook
title: Production Deploy Runbook
category: Engineering
owner: Platform Team
effective_date: 2025-08-01
status: current
---

# Production Deploy Runbook

## 1. When to deploy
Deploys to production happen on weekdays between 10:00 and 15:00 Central Time. A **deploy freeze starts 72 hours before every scheduled release** and during declared incidents. Freeze exceptions need the on-call manager's approval in #deploys.

## 2. Pre-flight
1. The change has 2 approving review(s) and a green CI pipeline.
2. The feature flag for any user-facing change is in the off state.
3. The deployer posts the change summary in #deploys.

## 3. Rollout
Deploys go through the canary stage (5 percent of traffic) for ten minutes, then to full rollout. Watch the error rate and p95 latency dashboards during the canary.

## 4. Rollback window
A deploy can be rolled back with a single click in the deploy console for **45 minutes** after full rollout completes. After that window, rollback requires a new deploy of the previous version following the Rollback Procedure.

## 5. After the deploy
Post the outcome in #deploys. Flip the feature flag per the Feature Flag Process.
