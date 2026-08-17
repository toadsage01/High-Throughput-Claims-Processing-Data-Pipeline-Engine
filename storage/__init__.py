"""
Storage Module — Cassandra Integration
======================================
Provides functionality for loading processed parquet data into Apache Cassandra.
The data is stored in the 'claims_audit' keyspace across five tables optimized
for different query patterns:

  - claims_by_id: single-claim lookups by claim_id
  - claims_by_provider: time-range queries per provider
  - provider_peer_stats: peer-group statistics by specialty + provider
  - beneficiary_profile: beneficiary summaries
  - flagged_claims: audit-flagged claims with score-based clustering
"""
