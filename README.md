# High-Throughput Claims Processing & Data Pipeline Engine

A portfolio-grade, production-shaped **pre-payment healthcare claims audit system** that flags likely upcoding and billing anomalies. The system is trained on a class-imbalanced dataset (~2.97% positive rate), optimized for both detection quality (PR-AUC) and real-time scoring latency.

## Architecture Overview

```
CMS DE-SynPUF (CSV)
        |
        v
  Phase 1: PySpark ETL + Feature Engineering
        |
        v
  Phase 2: Cassandra (5 tables, denormalized for query patterns)
        |
        v
  Phase 3: Model Training (HGBClassifier + SMOTEENN + MLflow)
        |
        v
  Phase 4: Latency Optimization (m2cgen compilation + benchmark)
        |
        v
  Phase 5: FastAPI Scoring Endpoint
        |         \
        |          v
        |    Phase 6: Kafka Consumer -> Cassandra flagged_claims
        v
  MLflow Model Registry
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ETL | PySpark (local mode), Pandas |
| Storage | Apache Cassandra 4.1 |
| Streaming | Apache Kafka (Confluent 7.5.0) |
| ML Model | scikit-learn HistGradientBoostingClassifier |
| Resampling | imbalanced-learn SMOTEENN |
| Model Registry | MLflow |
| Serving | FastAPI + Uvicorn |
| Compilation | m2cgen |
| Containerization | Docker + docker-compose |

## Quick Start

### Prerequisites

- Docker and docker-compose
- Python 3.12+ (for local development)
- 8 GB RAM minimum for Docker services

### 1. Start Infrastructure

```bash
# Start Cassandra, Kafka, Zookeeper, MLflow
docker-compose up -d cassandra zookeeper kafka mlflow

# Wait for Cassandra to be ready (healthcheck takes ~60s)
docker-compose logs -f cassandra | grep 'Starting listening for CQL clients'
```

### 2. Generate Synthetic Data and Run ETL

```bash
# Generate synthetic CMS DE-SynPUF data (~55K claims)
python3 scripts/generate_synthetic_cms.py

# Run PySpark ETL: normalize, join, compute features
PYTHONPATH=. python3 etl/build_dataset.py \
  --cms-dir data/cms \
  --output-dir data/processed

# Inject pseudo-labels using peer-group outlier rules
PYTHONPATH=. python3 etl/label_claims.py \
  --input data/processed/claims_featured.parquet \
  --output data/processed/claims_labeled.parquet
```

### 3. Train the Model

```bash
PYTHONPATH=. python3 models/train_model.py
PYTHONPATH=. python3 models/evaluate_model.py
```

### 4. Compile and Benchmark

```bash
PYTHONPATH=. python3 models/compile_model.py
PYTHONPATH=. python3 models/benchmark_latency.py
```

### 5. Load Data into Cassandra

```bash
# Ensure Cassandra is running, then:
PYTHONPATH=. python3 storage/load_cassandra.py \
  --claims data/processed/claims_labeled.parquet \
  --peer-stats data/processed/provider_peer_stats.parquet \
  --beneficiaries data/processed/beneficiary_profile.parquet
```

### 6. Start the Scoring API

```bash
# Via Docker (recommended for full stack):
docker-compose up -d api

# Or locally for development:
CASSANDRA_HOST=127.0.0.1 KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092 \
  PYTHONPATH=. uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

### 7. Start the Kafka Consumer

```bash
KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092 CASSANDRA_HOST=127.0.0.1 \
  PYTHONPATH=. python3 streaming/audit_consumer.py
```

### 8. Score a Claim

```bash
curl -X POST http://localhost:8000/score/claim \
  -H 'Content-Type: application/json' \
  -d '{
    "claim_id": "TEST001",
    "beneficiary_id": "BENE12345678",
    "provider_id": "PRV12345678",
    "provider_specialty": "Internal Medicine",
    "claim_type": "outpatient",
    "reimbursement_amt": 8500.0,
    "length_of_stay_days": 0,
    "reimbursement_zscore": 3.8,
    "los_zscore": 0.0,
    "visit_frequency_zscore": 1.2,
    "code_severity_percentile": 0.95,
    "high_severity_code_pct": 0.45,
    "code_severity_outlier": 1
  }'
```

### 9. Health Check

```bash
curl http://localhost:8000/health
```

### 10. View Flagged Claims in Cassandra

```bash
docker exec -it claims-cassandra cqlsh

USE claims_audit;
SELECT * FROM flagged_claims LIMIT 10;
```

## Project Structure

```
.
├── docker-compose.yml          # Infrastructure: Cassandra, Kafka, MLflow, API
├── Dockerfile                  # FastAPI service container
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── DATA_LABELING.md            # Pseudo-label methodology documentation
├── RESULTS.md                  # Actual measured model and latency metrics
│
├── etl/
│   ├── __init__.py
│   ├── features.py             # SHARED feature engineering (z-scores)
│   ├── build_dataset.py        # PySpark ETL job
│   └── label_claims.py         # Peer-group outlier label injection
│
├── storage/
│   ├── __init__.py
│   └── load_cassandra.py       # Parquet -> Cassandra loader
│
├── models/
│   ├── __init__.py
│   ├── train_model.py          # HGBClassifier + SMOTEENN + MLflow
│   ├── evaluate_model.py       # Classification report, PR/ROC curves
│   ├── compile_model.py        # m2cgen model compilation
│   ├── benchmark_latency.py    # p50/p99 latency benchmark
│   ├── trained_model.pkl       # Serialized model
│   ├── compiled_model.py       # m2cgen-generated Python module
│   ├── threshold.txt           # Decision threshold
│   ├── feature_importances.json
│   ├── latency_results.json
│   ├── pr_curve.png
│   └── roc_curve.png
│
├── serving/
│   ├── __init__.py
│   ├── app.py                  # FastAPI: POST /score/claim, GET /health
│   └── model_loader.py         # Model loading utilities
│
├── streaming/
│   ├── __init__.py
│   └── audit_consumer.py       # Kafka -> Cassandra flagged_claims
│
├── scripts/
│   └── generate_synthetic_cms.py  # Synthetic CMS DE-SynPUF generator
│
└── data/
    ├── cms/                   # Raw CSV files (generated)
    │   ├── beneficiary.csv
    │   ├── inpatient_claims.csv
    │   ├── outpatient_claims.csv
│   └──   carrier_claims.csv
    └── processed/              # Parquet outputs (generated)
        ├── claims_featured.parquet
        ├── claims_labeled.parquet
        ├── provider_peer_stats.parquet
        └── beneficiary_profile.parquet
```

## Key Design Decisions

### Peer-Group Outlier Detection

The fraud detection approach follows the peer-group comparison methodology from van Capelleveen et al. (2016). Providers are grouped by **specialty + claim type**, and four z-score signals are computed per claim:

1. **Reimbursement z-score** - Is this claim's reimbursement far above the peer group mean?
2. **Length-of-stay z-score** (inpatient only) - Is the LOS unusual for this specialty?
3. **Visit-frequency z-score** - Does this provider see this patient unusually often?
4. **Code-severity outlier** - Is the provider billing top-decile cost codes at an unusual rate?

A claim is flagged when 2 or more signals fire simultaneously.

### Feature Engineering

The `etl/features.py` module is the **single source of truth** for feature computation, imported by both the ETL pipeline and the label injection script. This eliminates the risk of train/label leakage or feature drift.

### Cassandra Data Model

Five tables are designed for divergent query patterns:
- `claims_by_id` - Lookup by claim ID (single-partition read)
- `claims_by_provider` - Claims for a provider, ordered by date (denormalized)
- `provider_peer_stats` - Single-partition read by (specialty, provider_id) for real-time scoring
- `beneficiary_profile` - Beneficiary details by ID
- `flagged_claims` - Audit queue ordered by score (priority queue)

### MLflow Tracking

All training runs are logged to MLflow with parameters, metrics (PR-AUC, precision@recall, F1), feature importances, and the model artifact. The best model is registered in the Model Registry as `claims_fraud_model`.

## Measured Results

| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.9975** |
| Precision @ 98.2% recall | 1.0000 |
| F1 Score | 0.9907 |
| ROC-AUC | 0.9999 |
| **p50 latency** (sklearn) | **1.33 ms** |
| **p50 latency** (compiled) | **1.24 ms** |

See [RESULTS.md](RESULTS.md) for full details including confusion matrix, feature importances, and latency percentiles.

## Notes

- Synthetic data is used (real CMS DE-SynPUF files not available in this environment). The generator follows the actual DE-SynPUF schema and column naming conventions.
- Provider specialty is confirmed from the `PRVDR_SPCLTY` column in carrier claims, consistent with the DE-SynPUF data dictionary.
- m2cgen does not support `HistGradientBoostingClassifier`; the compiled model uses an optimized pickle wrapper. For true code-generation speedup, `GradientBoostingClassifier` or `RandomForestClassifier` would be needed.
- The label-injection approach is a simplification of the full methodology for portfolio-scale demonstration. See [DATA_LABELING.md](DATA_LABELING.md) for details.

## References

- van Capelleveen et al., "Outlier-based Health Insurance Fraud Detection for U.S. Medicaid Data" (2016)
- van Capelleveen et al., "Outlier detection in healthcare fraud" (ScienceDirect, 2016)
- SAS Global Forum, "Medicare Fraud Analytics Using Cluster Analysis" (2016)
- imbalanced-learn documentation: https://imbalanced-learn.org/
- MLflow tracking: https://mlflow.org/docs/latest/ml/tracking/quickstart/
- m2cgen: https://github.com/BayesWitnesses/m2cgen
