---
doc_id: eng-logging-standards
title: Logging Standards
category: Engineering
owner: Platform Team
effective_date: 2024-04-01
status: current
---

# Logging Standards

Log in structured JSON with a timestamp, service name, request id and severity. Never log secrets, tokens, full card numbers or personal data beyond an opaque user id. Use INFO for business events, WARN for recoverable problems, and ERROR only for failures that need a human. Logs are retained as described in the Data Retention Policy. Correlate across services using the request id propagated in the `x-request-id` header.
