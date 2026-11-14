---
doc_id: eng-api-style-guide
title: API Style Guide
category: Engineering
owner: CTO Office
effective_date: 2024-08-01
status: current
---

# API Style Guide

Resources are plural nouns, identifiers are opaque strings, timestamps are RFC 3339 in UTC. Breaking changes require a new version prefix and a six-month deprecation period for the old one. Every endpoint documents its error shape, which is always `{"error": {"code": ..., "message": ...}}`. Paginate lists with an opaque cursor, never with page numbers. Rate limits are returned in standard headers.
