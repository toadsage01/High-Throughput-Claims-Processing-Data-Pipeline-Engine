"""
FastAPI Claims Scoring Application
===================================
Real-time REST API for scoring individual claims against the fraud/abuse
detection model.

Endpoints:
  POST /score/claim  — Accept a claim JSON payload, compute risk score,
                       return score + boolean flag.  If flagged, publish
                       to Kafka topic 'flagged-audits'.
  GET  /health       — Liveness / readiness probe.

Kafka Integration:
  When a claim's score meets or exceeds the decision threshold, a
  message is published to the 'flagged-audits' topic containing:
    claim_id, provider_id, score, driver_features (top 3).

Cassandra Integration:
  On each scoring request, the provider's peer-group stats are fetched
  from the provider_peer_stats table (single-partition read by
  specialty + provider_id) to provide context.

Middleware:
  A request-logging middleware logs every inbound request with
  method, path, status code, and duration.
"""

import json
import logging
import os
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from cassandra.cluster import Cluster
from kafka import KafkaProducer

from serving.model_loader import get_feature_names, get_threshold, load_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (environment-driven, Docker-friendly)
# ---------------------------------------------------------------------------

CASSANDRA_HOST = os.environ.get("CASSANDRA_HOST", "127.0.0.1")
CASSANDRA_PORT = int(os.environ.get("CASSANDRA_PORT", "9042"))
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = "flagged-audits"
KEYSPACE = "claims_audit"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Claims Audit Scoring API",
    description="Real-time fraud/abuse scoring for healthcare claims",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Lazy-loaded connections
# ---------------------------------------------------------------------------

_cassandra_session = None
_kafka_producer = None
_model = None
_threshold = None
_feature_names = None


def _get_cassandra_session():
    """Return (or create) the Cassandra session singleton."""
    global _cassandra_session
    if _cassandra_session is None:
        logger.info("Connecting to Cassandra at %s:%d ...", CASSANDRA_HOST, CASSANDRA_PORT)
        cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT)
        _cassandra_session = cluster.connect()
        _cassandra_session.execute(f"USE {KEYSPACE}")
        logger.info("Cassandra connection established.")
    return _cassandra_session


def _get_kafka_producer() -> KafkaProducer:
    """Return (or create) the Kafka producer singleton."""
    global _kafka_producer
    if _kafka_producer is None:
        logger.info("Connecting to Kafka at %s ...", KAFKA_BOOTSTRAP_SERVERS)
        _kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3,
            linger_ms=10,
        )
        logger.info("Kafka producer created.")
    return _kafka_producer


def _get_model():
    """Return the loaded model (or None if not yet compiled)."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _get_threshold() -> float:
    """Return the decision threshold (cached after first load)."""
    global _threshold
    if _threshold is None:
        _threshold = get_threshold()
    return _threshold


def _get_feature_names() -> List[str]:
    """Return the expected feature column names (cached)."""
    global _feature_names
    if _feature_names is None:
        _feature_names = get_feature_names()
    return _feature_names


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    """Pre-load model and threshold on application startup."""
    model = _get_model()
    if model is None:
        logger.warning("Model not available — scoring will use a placeholder.")
    threshold = _get_threshold()
    features = _get_feature_names()
    logger.info("Startup: threshold=%.4f, features=%s", threshold, features)


@app.on_event("shutdown")
async def on_shutdown():
    """Clean up connections on shutdown."""
    global _kafka_producer, _cassandra_session
    if _kafka_producer is not None:
        _kafka_producer.flush()
        _kafka_producer.close()
        logger.info("Kafka producer closed.")
    if _cassandra_session is not None:
        _cassandra_session.cluster.shutdown()
        logger.info("Cassandra connection closed.")


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status code, and elapsed time for every request."""
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ScoreClaimRequest(BaseModel):
    """JSON payload for the POST /score/claim endpoint.

    Fields mirror the claim schema produced by Phase 1 ETL, including
    the z-score features and code-severity features used by the model.
    Only the feature columns are required for scoring; additional fields
    are passed through for Kafka messaging.
    """
    claim_id: str = Field(..., description="Unique claim identifier")
    beneficiary_id: str = Field(..., description="Beneficiary DESYNPUF ID")
    provider_id: str = Field(..., description="Provider PRVDR_NUM")
    provider_specialty: str = Field(..., description="Provider specialty")
    claim_type: str = Field(..., description="inpatient | outpatient | carrier")
    claim_start_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    reimbursement_amt: float = Field(0.0)
    length_of_stay_days: int = Field(0)
    # Model features (z-scores)
    reimbursement_zscore: float = Field(0.0)
    los_zscore: float = Field(0.0)
    visit_frequency_zscore: float = Field(0.0)
    code_severity_percentile: float = Field(0.0)
    high_severity_code_pct: float = Field(0.0)
    code_severity_outlier: int = Field(0)
    # Optional extras
    diagnosis_codes: Optional[List[str]] = Field(default_factory=list)
    procedure_codes: Optional[List[str]] = Field(default_factory=list)


class ScoreClaimResponse(BaseModel):
    """Response for the scoring endpoint."""
    claim_id: str
    score: float
    flagged: bool
    threshold: float
    provider_peer_stats: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""
    status: str
    cassandra: str
    kafka: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Helper: extract top-N driver features
# ---------------------------------------------------------------------------

# Human-readable names for the feature columns, used in driver_features output
_FEATURE_DISPLAY_NAMES = {
    "reimbursement_zscore": "Reimbursement z-score",
    "los_zscore": "Length-of-stay z-score",
    "visit_frequency_zscore": "Visit-frequency z-score",
    "code_severity_percentile": "Code severity percentile",
    "high_severity_code_pct": "High-severity code %",
    "code_severity_outlier": "Code severity outlier",
}


def _top_driver_features(claim: ScoreClaimRequest, n: int = 3) -> List[str]:
    """Identify the top N features that drove the score.

    Uses a simple heuristic: rank by absolute z-score or percentile value.
    This is a placeholder — a true SHAP-based approach would be added in
    Phase 4.

    Args:
        claim: the incoming claim payload
        n: number of top features to return

    Returns:
        list of feature display names, sorted by descending contribution
    """
    feature_values = {
        "reimbursement_zscore": abs(claim.reimbursement_zscore),
        "los_zscore": abs(claim.los_zscore),
        "visit_frequency_zscore": abs(claim.visit_frequency_zscore),
        "code_severity_percentile": claim.code_severity_percentile,
        "high_severity_code_pct": claim.high_severity_code_pct / 100.0,
        "code_severity_outlier": float(claim.code_severity_outlier),
    }
    # Sort by absolute value, take top N
    sorted_features = sorted(feature_values.items(), key=lambda x: x[1], reverse=True)
    return [
        _FEATURE_DISPLAY_NAMES.get(f, f)
        for f, _ in sorted_features[:n]
    ]


# ---------------------------------------------------------------------------
# Helper: fetch provider peer stats from Cassandra
# ---------------------------------------------------------------------------

def _fetch_provider_peer_stats(specialty: str, provider_id: str) -> Optional[Dict[str, Any]]:
    """Fetch peer-group stats for a provider from Cassandra.

    Performs a single-partition read on provider_peer_stats using
    (specialty, provider_id) as the partition key.

    Args:
        specialty: provider specialty (partition key component)
        provider_id: provider ID (clustering column / second PK component)

    Returns:
        dict of peer stats, or None if not found / error
    """
    try:
        session = _get_cassandra_session()
        query = f"""
            SELECT provider_specialty, provider_id, claim_type,
                   n_claims, avg_reimbursement, reimbursement_stddev,
                   avg_length_of_stay, los_stddev,
                   avg_visit_freq_per_bene, peer_avg_visit_freq,
                   peer_visit_freq_stddev
            FROM {KEYSPACE}.provider_peer_stats
            WHERE provider_specialty = ? AND provider_id = ?
        """
        rows = session.execute(query, (specialty, provider_id))
        results = list(rows)
        if not results:
            logger.info("No peer stats found for specialty=%s, provider=%s",
                        specialty, provider_id)
            return None
        # Return the first row as a dict
        row = results[0]
        return {
            "provider_specialty": row.provider_specialty,
            "provider_id": row.provider_id,
            "claim_type": row.claim_type,
            "n_claims": row.n_claims,
            "avg_reimbursement": row.avg_reimbursement,
            "reimbursement_stddev": row.reimbursement_stddev,
            "avg_length_of_stay": row.avg_length_of_stay,
            "los_stddev": row.los_stddev,
            "avg_visit_freq_per_bene": row.avg_visit_freq_per_bene,
            "peer_avg_visit_freq": row.peer_avg_visit_freq,
            "peer_visit_freq_stddev": row.peer_visit_freq_stddev,
        }
    except Exception as exc:
        logger.error("Failed to fetch peer stats from Cassandra: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Helper: publish flagged claim to Kafka
# ---------------------------------------------------------------------------

def _publish_flagged_claim(claim_id: str, provider_id: str,
                            score: float, driver_features: List[str]):
    """Publish a flagged claim event to the 'flagged-audits' Kafka topic.

    Args:
        claim_id: unique claim identifier
        provider_id: provider responsible for the claim
        score: the model's risk score
        driver_features: list of top contributing feature names
    """
    try:
        producer = _get_kafka_producer()
        message = {
            "claim_id": claim_id,
            "provider_id": provider_id,
            "score": score,
            "driver_features": driver_features,
            "flag_date": date.today().isoformat(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        future = producer.send(KAFKA_TOPIC, value=message)
        # Block briefly to catch send errors
        future.get(timeout=5)
        logger.info(
            "Published flagged claim %s (score=%.4f) to topic '%s'",
            claim_id, score, KAFKA_TOPIC,
        )
    except Exception as exc:
        logger.error("Failed to publish to Kafka: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/score/claim", response_model=ScoreClaimResponse)
async def score_claim(claim: ScoreClaimRequest):
    """Score a single claim and return the risk score + flag.

    Steps:
      1. Build the feature vector from the request payload.
      2. Run the compiled model (or placeholder if model is unavailable).
      3. Compare score against the decision threshold.
      4. Fetch provider peer stats from Cassandra for context.
      5. If flagged, publish to Kafka 'flagged-audits' topic.
      6. Return score, flag, threshold, and optional peer stats.
    """
    model = _get_model()
    threshold = _get_threshold()
    features = _get_feature_names()

    # --- Build feature vector ---
    feature_vector = [
        getattr(claim, fname, 0.0) for fname in features
    ]

    # --- Score ---
    if model is not None:
        try:
            score = float(model.predict(feature_vector))
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Model inference error: {exc}")
    else:
        # Placeholder: use a simple heuristic score when model is not compiled
        # This allows the API to function end-to-end during development
        score = (
            abs(claim.reimbursement_zscore) * 0.3
            + abs(claim.los_zscore) * 0.2
            + abs(claim.visit_frequency_zscore) * 0.2
            + claim.code_severity_percentile * 0.15
            + claim.high_severity_code_pct / 100.0 * 0.1
            + float(claim.code_severity_outlier) * 0.05
        )
        logger.info("Using placeholder heuristic score (model not compiled)")

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))
    flagged = score >= threshold

    # --- Fetch provider peer stats from Cassandra ---
    peer_stats = _fetch_provider_peer_stats(claim.provider_specialty, claim.provider_id)

    # --- Kafka: publish if flagged ---
    if flagged:
        driver_features = _top_driver_features(claim, n=3)
        _publish_flagged_claim(
            claim_id=claim.claim_id,
            provider_id=claim.provider_id,
            score=score,
            driver_features=driver_features,
        )

    logger.info(
        "Claim %s scored: %.4f (threshold=%.4f, flagged=%s)",
        claim.claim_id, score, threshold, flagged,
    )

    return ScoreClaimResponse(
        claim_id=claim.claim_id,
        score=round(score, 6),
        flagged=flagged,
        threshold=threshold,
        provider_peer_stats=peer_stats,
        message=None if model is not None else "Model not compiled; using heuristic score.",
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health-check endpoint for orchestration and monitoring.

    Checks connectivity to Cassandra and Kafka, and whether the model
    has been loaded.
    """
    cassandra_ok = "ok"
    try:
        session = _get_cassandra_session()
        session.execute("SELECT release_version FROM system.local")
    except Exception as exc:
        cassandra_ok = f"error: {exc}"

    kafka_ok = "ok"
    try:
        producer = _get_kafka_producer()
        producer.partitions_for(KAFKA_TOPIC)
    except Exception as exc:
        kafka_ok = f"error: {exc}"

    return HealthResponse(
        status="healthy",
        cassandra=cassandra_ok,
        kafka=kafka_ok,
        model_loaded=_get_model() is not None,
    )
