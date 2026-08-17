"""
Cassandra Data Loader
====================
Reads processed parquet files and loads them into Apache Cassandra tables.

Creates the 'claims_audit' keyspace and five tables with schemas optimized
for the claims audit use case:

  1. claims_by_id           — O(1) claim lookup by claim_id
  2. claims_by_provider     — time-range scans per provider
  3. provider_peer_stats    — peer-group statistics per specialty/provider
  4. beneficiary_profile    — beneficiary summary by beneficiary_id
  5. flagged_claims         — flagged claims with score-desc clustering

Handles list-type columns (diagnosis_codes, procedure_codes, label_reasons,
chronic_conditions) by serializing them to JSON strings, since Cassandra's
native list type requires careful handling with the Python driver.

Usage:
    python -m storage.load_cassandra

    # Or with custom contact points:
    CASSANDRA_HOST=127.0.0.1 python -m storage.load_cassandra

Note:
    This script requires a running Cassandra instance. If Cassandra is not
    reachable, it will log a clear error message and exit gracefully.
"""

import json
import logging
import os
import sys
from datetime import datetime, date

import pandas as pd
from cassandra.cluster import Cluster, NoHostAvailable
from cassandra.auth import PlainTextAuthProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
CASSANDRA_HOST = os.environ.get("CASSANDRA_HOST", "127.0.0.1")
CASSANDRA_PORT = int(os.environ.get("CASSANDRA_PORT", "9042"))
KEYSPACE = "claims_audit"
REPLICATION = {"class": "SimpleStrategy", "replication_factor": 1}

# Parquet file paths (relative to project root)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CLAIMS_PARQUET = os.path.join(DATA_DIR, "claims_labeled.parquet")
PEER_STATS_PARQUET = os.path.join(DATA_DIR, "provider_peer_stats.parquet")
BENEFICIARY_PARQUET = os.path.join(DATA_DIR, "beneficiary_profile.parquet")

# Batch size for INSERT operations — keeps memory bounded
BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Keyspace & Table DDL
# ---------------------------------------------------------------------------

CREATE_KEYSPACE_CQL = f"""
CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
WITH replication = {json.dumps(REPLICATION)}
AND durable_writes = true;
"""

CREATE_TABLES_CQL = {
    # 1. Single-claim lookup
    "claims_by_id": f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE}.claims_by_id (
        claim_id            TEXT PRIMARY KEY,
        beneficiary_id      TEXT,
        provider_id         TEXT,
        provider_specialty  TEXT,
        claim_type          TEXT,
        claim_start_date    DATE,
        claim_end_date      DATE,
        claim_year          INT,
        admit_diagnosis_cd  TEXT,
        diagnosis_codes     TEXT,   -- JSON-serialized list
        procedure_codes     TEXT,   -- JSON-serialized list
        reimbursement_amt   DOUBLE,
        length_of_stay_days INT,
        reimbursement_zscore  DOUBLE,
        los_zscore           DOUBLE,
        visit_frequency_zscore DOUBLE,
        code_severity_percentile DOUBLE,
        high_severity_code_pct  DOUBLE,
        code_severity_outlier   INT,
        label                INT,
        label_reasons        TEXT    -- JSON-serialized list
    );
    """,

    # 2. Provider-scoped time-range queries
    "claims_by_provider": f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE}.claims_by_provider (
        provider_id         TEXT,
        claim_start_date    DATE,
        claim_id            TEXT,
        beneficiary_id      TEXT,
        provider_specialty  TEXT,
        claim_type          TEXT,
        claim_end_date      DATE,
        admit_diagnosis_cd  TEXT,
        diagnosis_codes     TEXT,
        procedure_codes     TEXT,
        reimbursement_amt   DOUBLE,
        length_of_stay_days INT,
        reimbursement_zscore  DOUBLE,
        los_zscore           DOUBLE,
        visit_frequency_zscore DOUBLE,
        code_severity_percentile DOUBLE,
        high_severity_code_pct  DOUBLE,
        code_severity_outlier   INT,
        label                INT,
        label_reasons        TEXT,
        PRIMARY KEY ((provider_id), claim_start_date, claim_id)
    ) WITH CLUSTERING ORDER BY (claim_start_date ASC, claim_id ASC);
    """,

    # 3. Peer-group statistics
    "provider_peer_stats": f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE}.provider_peer_stats (
        provider_specialty        TEXT,
        provider_id               TEXT,
        claim_type                TEXT,
        n_claims                  INT,
        avg_reimbursement         DOUBLE,
        reimbursement_stddev      DOUBLE,
        avg_length_of_stay        DOUBLE,
        los_stddev                DOUBLE,
        avg_visit_freq_per_bene   DOUBLE,
        peer_avg_visit_freq       DOUBLE,
        peer_visit_freq_stddev    DOUBLE,
        PRIMARY KEY ((provider_specialty), provider_id)
    );
    """,

    # 4. Beneficiary profiles
    "beneficiary_profile": f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE}.beneficiary_profile (
        beneficiary_id       TEXT PRIMARY KEY,
        birth_year           INT,
        chronic_conditions   TEXT,   -- JSON-serialized list
        claim_count_ytd      INT
    );
    """,

    # 5. Flagged claims (written by the Kafka consumer in Phase 6)
    "flagged_claims": f"""
    CREATE TABLE IF NOT EXISTS {KEYSPACE}.flagged_claims (
        flag_date           DATE,
        score               DOUBLE,
        claim_id            TEXT,
        provider_id         TEXT,
        driver_features     TEXT,   -- JSON-serialized list of top-3 features
        PRIMARY KEY ((flag_date), score, claim_id)
    ) WITH CLUSTERING ORDER BY (score DESC, claim_id ASC);
    """,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_date(val):
    """Convert various date representations to a Python date or None.

    Handles: datetime objects, date objects, 'YYYY-MM-DD' strings,
    pandas Timestamps, and NaT/None.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.date()
    if isinstance(val, str):
        try:
            return datetime.strptime(val, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _serialize_list(val):
    """Serialize a list (or NaN/None) to a JSON string for Cassandra storage.

    Cassandra's native list type is not reliably round-tripped through
    the Python driver for all element types, so we store lists as JSON text.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "[]"
    if isinstance(val, list):
        return json.dumps(val)
    # Defensive: if it's a string that looks like a list
    return str(val)


def _safe_float(val, default=0.0):
    """Convert val to float, returning default on failure/NaN."""
    try:
        v = float(val)
        return v if v == v else default  # NaN check
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    """Convert val to int, returning default on failure/NaN."""
    try:
        v = int(float(val))
        return v if v == v else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Core loading functions
# ---------------------------------------------------------------------------

def create_schema(session):
    """Create the keyspace and all five tables.

    Args:
        session: Cassandra cluster session
    """
    logger.info("Creating keyspace '%s'...", KEYSPACE)
    session.execute(CREATE_KEYSPACE_CQL)
    # Set keyspace for subsequent DDL
    session.execute(f"USE {KEYSPACE};")

    for table_name, ddl in CREATE_TABLES_CQL.items():
        logger.info("Creating table %s.%s ...", KEYSPACE, table_name)
        session.execute(ddl)

    logger.info("Schema creation complete.")


def load_claims(session, df):
    """Load claim rows into both claims_by_id and claims_by_provider tables.

    Each claim is inserted into two tables to support two query patterns:
      - Lookup by claim_id
      - Range scan by (provider_id, claim_start_date)

    Args:
        session: Cassandra cluster session
        df: DataFrame of labeled claims
    """
    total = len(df)
    logger.info("Loading %d claims into claims_by_id and claims_by_provider...", total)

    # Prepare insert statements for both tables
    insert_by_id = session.prepare(f"""
        INSERT INTO {KEYSPACE}.claims_by_id (
            claim_id, beneficiary_id, provider_id, provider_specialty,
            claim_type, claim_start_date, claim_end_date, claim_year,
            admit_diagnosis_cd, diagnosis_codes, procedure_codes,
            reimbursement_amt, length_of_stay_days,
            reimbursement_zscore, los_zscore, visit_frequency_zscore,
            code_severity_percentile, high_severity_code_pct,
            code_severity_outlier, label, label_reasons
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)

    insert_by_provider = session.prepare(f"""
        INSERT INTO {KEYSPACE}.claims_by_provider (
            provider_id, claim_start_date, claim_id,
            beneficiary_id, provider_specialty, claim_type,
            claim_end_date, admit_diagnosis_cd, diagnosis_codes,
            procedure_codes, reimbursement_amt, length_of_stay_days,
            reimbursement_zscore, los_zscore, visit_frequency_zscore,
            code_severity_percentile, high_severity_code_pct,
            code_severity_outlier, label, label_reasons
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)

    loaded = 0
    for i, row in df.iterrows():
        start_date = _parse_date(row.get("claim_start_date"))
        end_date = _parse_date(row.get("claim_end_date"))
        diag_codes = _serialize_list(row.get("diagnosis_codes"))
        proc_codes = _serialize_list(row.get("procedure_codes"))
        label_reasons = _serialize_list(row.get("label_reasons"))

        values = (
            str(row.get("claim_id", "")),
            str(row.get("beneficiary_id", "")),
            str(row.get("provider_id", "")),
            str(row.get("provider_specialty", "")),
            str(row.get("claim_type", "")),
            start_date,
            end_date,
            _safe_int(row.get("claim_year")),
            str(row.get("admit_diagnosis_cd", "")),
            diag_codes,
            proc_codes,
            _safe_float(row.get("reimbursement_amt")),
            _safe_int(row.get("length_of_stay_days")),
            _safe_float(row.get("reimbursement_zscore")),
            _safe_float(row.get("los_zscore")),
            _safe_float(row.get("visit_frequency_zscore")),
            _safe_float(row.get("code_severity_percentile")),
            _safe_float(row.get("high_severity_code_pct")),
            _safe_int(row.get("code_severity_outlier")),
            _safe_int(row.get("label")),
            label_reasons,
        )

        # Insert into claims_by_id
        session.execute(insert_by_id, values)

        # Insert into claims_by_provider (reorder: provider_id, start_date, claim_id first)
        provider_values = (
            str(row.get("provider_id", "")),
            start_date,
            str(row.get("claim_id", "")),
            str(row.get("beneficiary_id", "")),
            str(row.get("provider_specialty", "")),
            str(row.get("claim_type", "")),
            end_date,
            str(row.get("admit_diagnosis_cd", "")),
            diag_codes,
            proc_codes,
            _safe_float(row.get("reimbursement_amt")),
            _safe_int(row.get("length_of_stay_days")),
            _safe_float(row.get("reimbursement_zscore")),
            _safe_float(row.get("los_zscore")),
            _safe_float(row.get("visit_frequency_zscore")),
            _safe_float(row.get("code_severity_percentile")),
            _safe_float(row.get("high_severity_code_pct")),
            _safe_int(row.get("code_severity_outlier")),
            _safe_int(row.get("label")),
            label_reasons,
        )
        session.execute(insert_by_provider, provider_values)

        loaded += 1
        if loaded % BATCH_SIZE == 0:
            logger.info("  Progress: %d / %d claims loaded", loaded, total)

    logger.info("Claims loading complete: %d / %d rows into each of 2 tables", loaded, total)


def load_provider_peer_stats(session, df):
    """Load provider peer-group statistics into the provider_peer_stats table.

    Args:
        session: Cassandra cluster session
        df: DataFrame of provider peer stats
    """
    total = len(df)
    logger.info("Loading %d provider peer stats rows...", total)

    insert_stmt = session.prepare(f"""
        INSERT INTO {KEYSPACE}.provider_peer_stats (
            provider_specialty, provider_id, claim_type,
            n_claims, avg_reimbursement, reimbursement_stddev,
            avg_length_of_stay, los_stddev,
            avg_visit_freq_per_bene, peer_avg_visit_freq,
            peer_visit_freq_stddev
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)

    loaded = 0
    for _, row in df.iterrows():
        session.execute(insert_stmt, (
            str(row.get("provider_specialty", "")),
            str(row.get("provider_id", "")),
            str(row.get("claim_type", "")),
            _safe_int(row.get("n_claims")),
            _safe_float(row.get("avg_reimbursement")),
            _safe_float(row.get("reimbursement_stddev")),
            _safe_float(row.get("avg_length_of_stay")),
            _safe_float(row.get("los_stddev")),
            _safe_float(row.get("avg_visit_freq_per_bene")),
            _safe_float(row.get("peer_avg_visit_freq")),
            _safe_float(row.get("peer_visit_freq_stddev")),
        ))
        loaded += 1
        if loaded % BATCH_SIZE == 0:
            logger.info("  Progress: %d / %d rows", loaded, total)

    logger.info("Provider peer stats loading complete: %d rows", loaded)


def load_beneficiary_profiles(session, df):
    """Load beneficiary profiles into the beneficiary_profile table.

    Args:
        session: Cassandra cluster session
        df: DataFrame of beneficiary profiles
    """
    total = len(df)
    logger.info("Loading %d beneficiary profiles...", total)

    insert_stmt = session.prepare(f"""
        INSERT INTO {KEYSPACE}.beneficiary_profile (
            beneficiary_id, birth_year, chronic_conditions, claim_count_ytd
        ) VALUES (?, ?, ?, ?)
    """)

    loaded = 0
    for _, row in df.iterrows():
        chronic = _serialize_list(row.get("chronic_conditions"))
        session.execute(insert_stmt, (
            str(row.get("beneficiary_id", "")),
            _safe_int(row.get("birth_year")),
            chronic,
            _safe_int(row.get("claim_count_ytd")),
        ))
        loaded += 1
        if loaded % BATCH_SIZE == 0:
            logger.info("  Progress: %d / %d profiles", loaded, total)

    logger.info("Beneficiary profiles loading complete: %d rows", loaded)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point: read parquets, connect to Cassandra, load all tables.

    Exits gracefully with a clear error message if Cassandra is not reachable.
    """
    logger.info("=" * 60)
    logger.info("Cassandra Data Loader — Phase 2")
    logger.info("=" * 60)

    # --- Read parquet files ---
    logger.info("Reading parquet files from %s ...", DATA_DIR)

    if not os.path.exists(CLAIMS_PARQUET):
        logger.error("Claims parquet not found: %s", CLAIMS_PARQUET)
        logger.error("Run the ETL pipeline first (Phase 1).")
        sys.exit(1)

    claims_df = pd.read_parquet(CLAIMS_PARQUET)
    logger.info("  claims_labeled.parquet: %d rows, columns: %s",
                len(claims_df), list(claims_df.columns))

    peer_stats_df = pd.read_parquet(PEER_STATS_PARQUET)
    logger.info("  provider_peer_stats.parquet: %d rows", len(peer_stats_df))

    bene_df = pd.read_parquet(BENEFICIARY_PARQUET)
    logger.info("  beneficiary_profile.parquet: %d rows", len(bene_df))

    # --- Connect to Cassandra ---
    logger.info("Connecting to Cassandra at %s:%d ...", CASSANDRA_HOST, CASSANDRA_PORT)

    try:
        cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT)
        session = cluster.connect()
    except NoHostAvailable as exc:
        logger.error(
            "Cannot connect to Cassandra at %s:%d. Is it running?\n"
            "  Error: %s\n"
            "  Start Cassandra with: docker-compose up -d cassandra\n"
            "  Or run locally and set CASSANDRA_HOST/CASSANDRA_PORT env vars.",
            CASSANDRA_HOST, CASSANDRA_PORT, exc,
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected error connecting to Cassandra: %s", exc)
        sys.exit(1)

    try:
        # --- Create schema ---
        create_schema(session)

        # --- Load data ---
        load_provider_peer_stats(session, peer_stats_df)
        load_beneficiary_profiles(session, bene_df)
        load_claims(session, claims_df)

        logger.info("=" * 60)
        logger.info("All data loaded successfully into Cassandra.")
        logger.info("=" * 60)

    finally:
        cluster.shutdown()
        logger.info("Cassandra connection closed.")


if __name__ == "__main__":
    main()
