---
doc_id: eng-observability-guide
title: Observability Guide
category: Engineering
owner: Platform Team
effective_date: 2025-02-10
status: current
---

# Observability Guide

Every service exposes request rate, error rate and latency percentiles on the standard dashboard template. Alerts page on symptoms (error rate, latency, saturation), not on causes. Each alert links to a runbook section. Dashboards are reviewed in the fortnightly platform meeting; unused panels are deleted. Traces are sampled at 10 percent and 100 percent for errors.
