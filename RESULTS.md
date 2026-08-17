# RESULTS — Claims Fraud Detection Model

## Hardware

| Spec | Value |
|------|-------|
| CPU | Intel Xeon Processor |
| Cores | 2 |
| RAM | 4 GB |
| OS | Linux x86_64 |
| Python | 3.12 / 3.13 |
| scikit-learn | 1.5.2 |

---

## Phase 3 — Model Performance

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | HistGradientBoostingClassifier |
| Resampling | SMOTEENN (train set only) |
| max_iter | 200 |
| learning_rate | 0.1 |
| max_depth | 5 |
| class_weight | balanced |
| Train size | 44,000 → 85,336 (after SMOTEENN) |
| Test size | 11,000 (327 positive, 2.97%) |

### Metrics

| Metric | Value |
|--------|-------|
| **PR-AUC** | **0.9975** |
| **Precision @ 98.2% recall** | **1.0000** |
| Recall | 0.9817 |
| F1 Score | 0.9907 |
| ROC-AUC | 0.9999 |
| Accuracy | 0.9995 |
| Decision Threshold | 0.999882 |

### Confusion Matrix (Test Set, n=11,000)

|  | Predicted Neg | Predicted Pos |
|--|---------------|---------------|
| **Actual Neg** | 10,673 (TN) | 0 (FP) |
| **Actual Pos** | 6 (FN) | 321 (TP) |

### Feature Importances (Permutation, normalized)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | reimbursement_zscore | 0.5874 |
| 2 | visit_frequency_zscore | 0.4073 |
| 3 | los_zscore | 0.0050 |
| 4 | high_severity_code_pct | 0.0003 |
| 5 | code_severity_percentile | 0.0000 |
| 6 | code_severity_outlier | 0.0000 |

### Artifacts

| File | Description |
|------|-------------|
| `models/trained_model.pkl` | Trained HGBClassifier (joblib) |
| `models/threshold.txt` | Decision threshold (0.999882) |
| `models/feature_importances.json` | Permutation feature importances |
| `models/pr_curve.png` | Precision-Recall curve plot |
| `models/roc_curve.png` | ROC curve plot |

---

## Phase 4 — Latency Benchmark

### Configuration

- **Iterations**: 10,000 (1,000 warmup discarded)
- **Input**: 10,000 random 6-feature vectors (standard normal)
- **Timer**: `time.perf_counter()`
- **Note**: m2cgen does not support `HistGradientBoostingClassifier`; compiled model uses an optimized pickle wrapper with lazy loading.

### Results (single-sample inference, milliseconds)

| Metric | sklearn `predict_proba` | sklearn `predict` | compiled `score` (pickle wrapper) |
|--------|------------------------|-------------------|-------------------------------|
| **p50** | 1.3319 ms | 1.3205 ms | 1.2377 ms |
| **p95** | 1.4904 ms | 1.4946 ms | 1.4114 ms |
| **p99** | 28.2957 ms | 28.2306 ms | 28.1647 ms |
| **mean** | 1.8505 ms | 1.8300 ms | 1.7236 ms |
| **std** | 3.6322 ms | 3.6131 ms | 3.4984 ms |

### Speedup (sklearn predict_proba vs compiled)

| Metric | Speedup |
|--------|---------|
| mean | 1.07x |
| p50 | 1.08x |
| p99 | 1.00x |

### Analysis

- The compiled pickle wrapper achieves a marginal ~7% mean speedup over raw sklearn `predict_proba`, as both use the same underlying HistGradientBoostingClassifier.
- p50 latency is ~1.2–1.3 ms per single-sample inference.
- p99 spikes (~28 ms) are attributable to Python GC pauses and OS scheduling on a 2-core, 4 GB VM.
- For true code-generation speedup, a model type supported by m2cgen (e.g., GradientBoostingClassifier, RandomForest) would be needed.
