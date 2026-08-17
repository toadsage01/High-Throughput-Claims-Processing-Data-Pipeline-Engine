#!/usr/bin/env python3
"""
Phase 4 — Latency Benchmarking
Benchmarks sklearn model.predict() vs compiled model score() latency.
Uses time.perf_counter() with 10,000 iterations, 1,000 warmup.
"""

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
COMPILED_PATH = os.path.join(MODEL_DIR, "compiled_model.py")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")
LATENCY_PATH = os.path.join(MODEL_DIR, "latency_results.json")

N_SAMPLES = 10_000
N_WARMUP = 1_000
N_ITERATIONS = 10_000


def run_benchmark(model, data, name, use_proba=True):
    """Run latency benchmark, return dict of results."""
    print(f"\n  Benchmarking: {name}")
    print(f"    Warmup: {N_WARMUP} calls …")

    # Warmup
    for i in range(N_WARMUP):
        row = data[i]
        if use_proba:
            _ = model.predict_proba(row.reshape(1, -1))
        else:
            _ = model.predict(row.reshape(1, -1))

    print(f"    Measuring {N_ITERATIONS} calls …")
    latencies = []
    for i in range(N_ITERATIONS):
        row = data[i]
        t0 = time.perf_counter()
        if use_proba:
            _ = model.predict_proba(row.reshape(1, -1))
        else:
            _ = model.predict(row.reshape(1, -1))
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    latencies = np.array(latencies)
    results = {
        "name": name,
        "n_iterations": N_ITERATIONS,
        "mean_ms": float(latencies.mean() * 1000),
        "std_ms": float(latencies.std() * 1000),
        "p50_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
        "p99_ms": float(np.percentile(latencies, 99) * 1000),
        "min_ms": float(latencies.min() * 1000),
        "max_ms": float(latencies.max() * 1000),
    }
    return results


def run_benchmark_compiled(score_fn, data, name):
    """Benchmark compiled model's score function."""
    print(f"\n  Benchmarking: {name}")
    print(f"    Warmup: {N_WARMUP} calls …")

    # Warmup
    for i in range(N_WARMUP):
        row = data[i].tolist()
        _ = score_fn(row)

    print(f"    Measuring {N_ITERATIONS} calls …")
    latencies = []
    for i in range(N_ITERATIONS):
        row = data[i].tolist()
        t0 = time.perf_counter()
        _ = score_fn(row)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    latencies = np.array(latencies)
    results = {
        "name": name,
        "n_iterations": N_ITERATIONS,
        "mean_ms": float(latencies.mean() * 1000),
        "std_ms": float(latencies.std() * 1000),
        "p50_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
        "p99_ms": float(np.percentile(latencies, 99) * 1000),
        "min_ms": float(latencies.min() * 1000),
        "max_ms": float(latencies.max() * 1000),
    }
    return results


def main():
    print("=" * 60)
    print("Phase 4 — Latency Benchmark")
    print("=" * 60)

    # Hardware info
    print("\nHardware: Intel Xeon Processor, 2 cores, 4GB RAM, Linux x86_64")

    # Load sklearn model
    print("\nLoading sklearn model …")
    sklearn_model = joblib.load(MODEL_PATH)

    # Load compiled model
    print("Loading compiled model …")
    import importlib.util
    spec = importlib.util.spec_from_file_location("compiled_model", COMPILED_PATH)
    compiled_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiled_mod)

    # Generate random feature vectors
    print(f"\nGenerating {N_SAMPLES:,} random feature vectors …")
    rng = np.random.RandomState(42)
    data = rng.randn(N_SAMPLES, 6).astype(np.float64)

    # ── sklearn predict_proba benchmark ──────────────────────────────────────
    sklearn_proba = run_benchmark(
        sklearn_model, data, "sklearn predict_proba", use_proba=True
    )
    print(f"    Mean: {sklearn_proba['mean_ms']:.4f} ms")
    print(f"    P50:  {sklearn_proba['p50_ms']:.4f} ms")
    print(f"    P95:  {sklearn_proba['p95_ms']:.4f} ms")
    print(f"    P99:  {sklearn_proba['p99_ms']:.4f} ms")

    # ── sklearn predict benchmark ─────────────────────────────────────────────
    sklearn_pred = run_benchmark(
        sklearn_model, data, "sklearn predict", use_proba=False
    )
    print(f"    Mean: {sklearn_pred['mean_ms']:.4f} ms")
    print(f"    P50:  {sklearn_pred['p50_ms']:.4f} ms")
    print(f"    P95:  {sklearn_pred['p95_ms']:.4f} ms")
    print(f"    P99:  {sklearn_pred['p99_ms']:.4f} ms")

    # ── compiled model benchmark ──────────────────────────────────────────────
    compiled_results = run_benchmark_compiled(
        compiled_mod.score, data, "compiled_model score (pickle wrapper)"
    )
    print(f"    Mean: {compiled_results['mean_ms']:.4f} ms")
    print(f"    P50:  {compiled_results['p50_ms']:.4f} ms")
    print(f"    P95:  {compiled_results['p95_ms']:.4f} ms")
    print(f"    P99:  {compiled_results['p99_ms']:.4f} ms")

    # ── Summary ───────────────────────────────────────────────────────────────
    speedup_mean = sklearn_proba["mean_ms"] / compiled_results["mean_ms"] if compiled_results["mean_ms"] > 0 else 0
    speedup_p50 = sklearn_proba["p50_ms"] / compiled_results["p50_ms"] if compiled_results["p50_ms"] > 0 else 0
    speedup_p99 = sklearn_proba["p99_ms"] / compiled_results["p99_ms"] if compiled_results["p99_ms"] > 0 else 0

    output = {
        "hardware": "Intel Xeon Processor, 2 cores, 4GB RAM, Linux x86_64",
        "n_samples": N_SAMPLES,
        "n_warmup": N_WARMUP,
        "n_iterations": N_ITERATIONS,
        "benchmarks": {
            "sklearn_predict_proba": sklearn_proba,
            "sklearn_predict": sklearn_pred,
            "compiled_model_score": compiled_results,
        },
        "speedup_sklearn_proba_vs_compiled": {
            "mean_x": round(speedup_mean, 2),
            "p50_x": round(speedup_p50, 2),
            "p99_x": round(speedup_p99, 2),
        },
        "note": "m2cgen does not support HistGradientBoostingClassifier; compiled_model.py uses optimized pickle wrapper with lazy loading.",
    }

    with open(LATENCY_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {LATENCY_PATH}")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"\n  sklearn predict_proba:")
    print(f"    p50={sklearn_proba['p50_ms']:.4f}ms  p95={sklearn_proba['p95_ms']:.4f}ms  p99={sklearn_proba['p99_ms']:.4f}ms  mean={sklearn_proba['mean_ms']:.4f}ms  std={sklearn_proba['std_ms']:.4f}ms")
    print(f"\n  sklearn predict:")
    print(f"    p50={sklearn_pred['p50_ms']:.4f}ms  p95={sklearn_pred['p95_ms']:.4f}ms  p99={sklearn_pred['p99_ms']:.4f}ms  mean={sklearn_pred['mean_ms']:.4f}ms  std={sklearn_pred['std_ms']:.4f}ms")
    print(f"\n  compiled_model score (pickle wrapper):")
    print(f"    p50={compiled_results['p50_ms']:.4f}ms  p95={compiled_results['p95_ms']:.4f}ms  p99={compiled_results['p99_ms']:.4f}ms  mean={compiled_results['mean_ms']:.4f}ms  std={compiled_results['std_ms']:.4f}ms")
    print(f"\n  Speedup (sklearn_proba / compiled): mean={speedup_mean:.2f}x  p50={speedup_p50:.2f}x  p99={speedup_p99:.2f}x")


if __name__ == "__main__":
    main()
