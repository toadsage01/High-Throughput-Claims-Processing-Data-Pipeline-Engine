"""
Streaming Module — Kafka Audit Consumer
========================================
Consumes flagged-claim events from the 'flagged-audits' Kafka topic
and persists them to the Cassandra flagged_claims table.

This closes the loop: FastAPI scores a claim -> publishes to Kafka ->
consumer writes to Cassandra for audit trail and dashboard queries.
"""
