---
doc_id: eng-disaster-recovery
title: Disaster Recovery Runbook
category: Engineering
owner: Platform Team
effective_date: 2025-05-20
status: current
---

# Disaster Recovery Runbook

## 1. Objectives
Recovery objectives are set per service and reviewed every six months. Definitions of RTO and RPO are in the Glossary.

| Service | RTO | RPO |
|---|---|---|
| Payments service | **4 hours** | 5 minutes |
| Orders database | 6 hours | **30 minutes** |
| Fleet telemetry | 12 hours | 1 hour |
| Internal dashboards | 48 hours | 24 hours |

## 2. Backups
Databases are backed up continuously to a second region with point-in-time recovery. Object storage is replicated cross-region. Backup restores are tested quarterly by the Platform Team and the results posted in #platform.

## 3. Regional failover
Failover to the secondary region is triggered by the incident commander for a SEV1 that is expected to exceed the affected service's RTO. The failover script lives in the platform-ops repository; run it from the bastion host. DNS changes propagate within five minutes.

## 4. Communication
Failover is announced in #sev1-bridge and on the status page. The CTO is informed for any failover affecting the payments service.
