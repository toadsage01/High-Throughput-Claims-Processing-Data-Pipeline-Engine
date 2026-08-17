#!/usr/bin/env python3
"""
Compiled model wrapper — loads optimized pickle and exposes a score() function.
Generated as fallback when m2cgen compilation was unavailable.
"""
import pickle
import os
import numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimized_model.pkl")
_model = None

def _get_model():
    global _model
    if _model is None:
        with open(_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model

def score(features):
    """
    Score a single claim.
    features: list or array of 6 numeric values in the order:
        [reimbursement_zscore, los_zscore, visit_frequency_zscore,
         code_severity_percentile, high_severity_code_pct, code_severity_outlier]
    Returns: probability of fraud (float).
    """
    model = _get_model()
    x = np.array(features, dtype=np.float64).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])
