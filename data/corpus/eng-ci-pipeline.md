---
doc_id: eng-ci-pipeline
title: CI Pipeline Guide
category: Engineering
owner: Platform Team
effective_date: 2024-12-01
status: current
---

# CI Pipeline Guide

The pipeline runs lint, unit tests, integration tests against ephemeral databases, and a container build on every pull request. Firmware repositories additionally run hardware-in-the-loop tests on the lab bench pool, which are queued and may take up to 40 minutes. Flaky tests are quarantined with a ticket, never retried silently. A red main branch is everyone's problem: the person whose change broke it fixes or reverts within an hour.
