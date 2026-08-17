"""
Serving Module — FastAPI Claims Scoring Service
===============================================
Provides the REST API for real-time claim scoring and flagging.

Endpoints:
  POST /score/claim  — Score a single claim, return risk score + flag
  GET  /health       — Health-check for orchestration

When a claim is flagged (score >= threshold), the result is published
to the Kafka topic 'flagged-audits' for downstream processing by
the streaming consumer (Phase 6).
"""
