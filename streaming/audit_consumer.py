"""
Kafka Audit Consumer — Flagged Claims to Cassandra
=====================================================
Consumes messages from the 'flagged-audits' Kafka topic and writes each
flagged-claim record into the Cassandra flagged_claims table.

This consumer forms the streaming layer (Phase 6) of the pipeline:
  FastAPI /score/claim  ->  Kafka 'flagged-audits'  ->  THIS consumer  ->  Cassandra

Message schema (produced by serving/app.py):
  {
    "claim_id": str,
    "provider_id": str,
    "score": float,
    "driver_features": [str, ...],  // top 3 contributing features
    "flag_date": "YYYY-MM-DD",
    "timestamp": ISO-8601
  }

Cassandra target table:
  flagged_claims (flag_date DATE, score DOUBLE, claim_id TEXT,
                   provider_id TEXT, driver_features TEXT,
                   PRIMARY KEY ((flag_date), score, claim_id))
  CLUSTERING ORDER BY (score DESC, claim_id ASC)

Shutdown:
  Handles SIGTERM and SIGINT for graceful shutdown inside Docker/K8s.

Usage:
    python -m streaming.audit_consumer

    # With custom bootstrap servers:
    KAFKA_BOOTSTRAP_SERVERS=kafka:9092 python -m streaming.audit_consumer
"""

import json
import logging
import os
import signal
import sys
from datetime import date
from typing import Any, Dict

from kafka import KafkaConsumer
from cassandra.cluster import Cluster, NoHostAvailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
CASSANDRA_HOST = os.environ.get("CASSANDRA_HOST", "127.0.0.1")
CASSANDRA_PORT = int(os.environ.get("CASSANDRA_PORT", "9042"))
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = "flagged-audits"
KAFKA_GROUP_ID = "audit-consumer-group"
KEYSPACE = "claims_audit"

# Graceful shutdown flag
_shutdown_requested = False


def _handle_signal(signum, frame):
    """Signal handler that sets the shutdown flag."""
    global _shutdown_requested
    logger.info("Received signal %d, initiating graceful shutdown...", signum)
    _shutdown_requested = True


def _parse_flag_date(val: Any) -> date:
    """Parse a flag_date value into a Python date object.

    Handles 'YYYY-MM-DD' strings and date objects.
    Falls back to today's date on failure.
    """
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            from datetime import datetime
            return datetime.strptime(val, "%Y-%m-%d").date()
        except ValueError:
            pass
    logger.warning("Could not parse flag_date '%s', using today.", val)
    return date.today()


def _connect_cassandra():
    """Establish a Cassandra connection and return a prepared insert statement.

    Returns:
        tuple: (session, prepared_statement)

    Raises:
        SystemExit: if Cassandra is not reachable
    """
    logger.info("Connecting to Cassandra at %s:%d ...", CASSANDRA_HOST, CASSANDRA_PORT)
    try:
        cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT)
        session = cluster.connect()
        session.execute(f"USE {KEYSPACE}")
        logger.info("Cassandra connection established.")
    except NoHostAvailable as exc:
        logger.error(
            "Cannot connect to Cassandra at %s:%d. Error: %s\n"
            "  Ensure Cassandra is running and the keyspace '%s' exists.",
            CASSANDRA_HOST, CASSANDRA_PORT, exc, KEYSPACE,
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected Cassandra error: %s", exc)
        sys.exit(1)

    # Prepare the INSERT statement once for reuse
    insert_stmt = session.prepare(f"""
        INSERT INTO {KEYSPACE}.flagged_claims (
            flag_date, score, claim_id, provider_id, driver_features
        ) VALUES (?, ?, ?, ?, ?)
    """)
    logger.info("Prepared INSERT statement for flagged_claims.")
    return session, insert_stmt, cluster


def _connect_kafka() -> KafkaConsumer:
    """Create and return a Kafka consumer for the flagged-audits topic.

    Returns:
        KafkaConsumer instance configured to read from the topic.
    """
    logger.info(
        "Connecting to Kafka at %s (topic: %s, group: %s) ...",
        KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_GROUP_ID,
    )
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=1000,  # poll timeout so we can check _shutdown_requested
    )
    logger.info("Kafka consumer created.")
    return consumer


def _process_message(msg_value: Dict[str, Any], session, insert_stmt):
    """Process a single Kafka message: validate and write to Cassandra.

    Args:
        msg_value: deserialized JSON message from Kafka
        session: active Cassandra session
        insert_stmt: prepared INSERT statement for flagged_claims
    """
    try:
        claim_id = msg_value.get("claim_id", "")
        provider_id = msg_value.get("provider_id", "")
        score = float(msg_value.get("score", 0.0))
        driver_features = msg_value.get("driver_features", [])
        flag_date = _parse_flag_date(msg_value.get("flag_date"))

        if not claim_id:
            logger.warning("Skipping message with empty claim_id: %s", msg_value)
            return

        # Serialize driver_features to JSON string for Cassandra TEXT column
        features_json = json.dumps(driver_features) if isinstance(driver_features, list) else str(driver_features)

        session.execute(insert_stmt, (
            flag_date,
            score,
            claim_id,
            provider_id,
            features_json,
        ))

        logger.info(
            "Wrote flagged claim to Cassandra: claim_id=%s, provider=%s, "
            "score=%.4f, date=%s, drivers=%s",
            claim_id, provider_id, score, flag_date.isoformat(), features_json,
        )

    except Exception as exc:
        logger.error("Failed to process message: %s | Error: %s", msg_value, exc)


def main():
    """Main loop: consume from Kafka, write to Cassandra, handle shutdown.

    Registers SIGTERM/SIGINT handlers for graceful shutdown inside
    Docker containers.  The consumer polls in a loop, writing each
    message to Cassandra, until a shutdown signal is received.
    """
    global _shutdown_requested

    logger.info("=" * 60)
    logger.info("Audit Consumer — Phase 6")
    logger.info("=" * 60)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Connect to Cassandra and Kafka
    session, insert_stmt, cluster = _connect_cassandra()
    consumer = _connect_kafka()

    messages_processed = 0
    errors = 0

    try:
        logger.info("Starting consumption loop...")
        while not _shutdown_requested:
            # poll() blocks for up to consumer_timeout_ms (1s)
            # Then returns an empty iterator if no messages
            for message in consumer:
                if _shutdown_requested:
                    break
                try:
                    _process_message(message.value, session, insert_stmt)
                    messages_processed += 1
                except Exception as exc:
                    logger.error("Unhandled error processing message: %s", exc)
                    errors += 1

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
        _shutdown_requested = True
    except Exception as exc:
        logger.error("Fatal error in consumer loop: %s", exc)
    finally:
        logger.info("Shutting down...")
        logger.info(
            "Totals: %d messages processed, %d errors.",
            messages_processed, errors,
        )
        consumer.close()
        logger.info("Kafka consumer closed.")
        cluster.shutdown()
        logger.info("Cassandra connection closed.")
        logger.info("Audit consumer stopped.")


if __name__ == "__main__":
    main()
