"""
Shared Feature Engineering Module
===============================
This module computes the z-score features used for BOTH label injection
and model training. It is imported by both etl/build_dataset.py (for
the training dataset) and etl/label_claims.py (for generating pseudo-labels).

Keeping the feature logic in ONE place prevents train/label leakage and
drift between the labeling step and the model training step.

Peer group definition: same provider_specialty + same claim_type.
This is critical because baseline cost and LOS differ enormously across
specialties and claim types.

Feature formulas (per claim, using provider's peer-group statistics):
  1. reimbursement_zscore: (claim_amt - peer_avg) / peer_stddev
  2. los_zscore: (claim_los - peer_avg_los) / peer_los_stddev (inpatient only)
  3. visit_frequency_zscore: (beneficiary_visits_to_provider_ytd - peer_avg_vf) / peer_vf_stddev
  4. code_severity_percentile: rank of claim's diagnosis/procedure codes by
     reimbursement weight within their code family
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Reimbursement weight lookup — maps diagnosis/procedure codes to
# approximate average reimbursement amounts. Built from the data itself.
# This is populated by compute_code_reimbursement_weights().


def compute_peer_group_stats(df):
    """Compute peer-group statistics for each (specialty, claim_type) group.
    
    These stats are used to compute z-scores for individual claims.
    Peer group = same provider_specialty + same claim_type.
    
    Args:
        df: DataFrame with columns [provider_id, provider_specialty,
            claim_type, reimbursement_amt, length_of_stay_days]
    
    Returns:
        DataFrame with one row per (specialty, claim_type, provider_id)
        containing mean/stddev for reimbursement, LOS, and visit frequency.
    """
    logger.info("Computing peer-group statistics...")
    
    # Visit frequency: count of visits per (beneficiary, provider) pair in the year
    visit_freq = (
        df.groupby(["beneficiary_id", "provider_id", "claim_year"])
        .size()
        .reset_index(name="visit_count")
    )
    
    # Average visit frequency per provider within each peer group
    # (i.e., how often does this provider see each patient, on average)
    provider_visit_freq = (
        visit_freq.groupby(["provider_id", "claim_year"])
        ["visit_count"].mean()
        .reset_index(name="avg_visit_freq_per_bene")
    )
    
    # Provider-level stats per peer group
    provider_stats = (
        df.groupby(["provider_specialty", "claim_type", "provider_id"])
        .agg(
            n_claims=("claim_id", "count"),
            avg_reimbursement=("reimbursement_amt", "mean"),
            reimbursement_stddev=("reimbursement_amt", "std"),
            avg_length_of_stay=("length_of_stay_days", "mean"),
            los_stddev=("length_of_stay_days", "std"),
        )
        .reset_index()
    )
    
    # Merge visit frequency stats
    provider_stats = provider_stats.merge(
        provider_visit_freq.rename(columns={"claim_year": "_drop"}),
        on="provider_id",
        how="left"
    ).drop(columns=["_drop"], errors="ignore")
    
    # Compute peer-group level stats (across all providers in the same specialty+claim_type)
    # For visit frequency, we compute mean/std across providers
    vf_by_peer_group = (
        provider_stats.groupby(["provider_specialty", "claim_type"])
        ["avg_visit_freq_per_bene"].agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "peer_avg_visit_freq", "std": "peer_visit_freq_stddev"})
    )
    
    provider_stats = provider_stats.merge(vf_by_peer_group, on=["provider_specialty", "claim_type"])
    
    # Replace zero/NaN stddevs with a small value to avoid division by zero
    for std_col in ["reimbursement_stddev", "los_stddev", "peer_visit_freq_stddev"]:
        provider_stats[std_col] = provider_stats[std_col].replace(0, np.nan)
        provider_stats[std_col] = provider_stats[std_col].fillna(
            provider_stats[std_col].median()
        )
    
    logger.info(f"Computed stats for {len(provider_stats)} providers across "
                f"{provider_stats['provider_specialty'].nunique()} specialties")
    
    return provider_stats


def compute_code_reimbursement_weights(df):
    """Build a lookup table mapping each code to its average reimbursement.
    
    This is used to compute the code-severity percentile: how expensive
    are this claim's codes relative to all claims using the same code?
    
    Args:
        df: DataFrame with columns [diagnosis_codes, procedure_codes, reimbursement_amt]
    
    Returns:
        dict mapping code -> (mean_reimbursement, count)
    """
    logger.info("Computing code reimbursement weights...")
    
    code_weights = defaultdict(lambda: [0.0, 0])  # [sum_reimb, count]
    
    for _, row in df.iterrows():
        amt = row["reimbursement_amt"]
        # Process diagnosis codes
        for code in row.get("diagnosis_codes", []):
            if code and code.strip():
                code_weights[code][0] += amt
                code_weights[code][1] += 1
        # Process procedure codes
        for code in row.get("procedure_codes", []):
            if code and code.strip():
                code_weights[code][0] += amt
                code_weights[code][1] += 1
    
    # Compute averages
    avg_weights = {
        code: (total / count, count)
        for code, (total, count) in code_weights.items()
        if count > 0
    }
    
    logger.info(f"Computed reimbursement weights for {len(avg_weights)} unique codes")
    return avg_weights


def compute_code_severity_percentile(diag_codes, proc_codes, code_weights):
    """Compute the severity percentile for a claim's codes.
    
    For each code on the claim, look up its average reimbursement weight.
    Return the maximum percentile across all codes (i.e., the most expensive
    code family used on this claim).
    
    Args:
        diag_codes: list of diagnosis code strings
        proc_codes: list of procedure code strings
        code_weights: dict from compute_code_reimbursement_weights()
    
    Returns:
        float: the highest severity percentile (0-1) among the claim's codes
    """
    if not code_weights:
        return 0.5  # default to median if no weights available
    
    all_avg_reimbs = [v[0] for v in code_weights.values()]
    if not all_avg_reimbs:
        return 0.5
    
    all_avg_reimbs = np.array(all_avg_reimbs)
    p50 = np.percentile(all_avg_reimbs, 50)
    p90 = np.percentile(all_avg_reimbs, 90)
    
    max_percentile = 0.0
    for code in list(diag_codes) + list(proc_codes):
        if code and code in code_weights:
            avg_reimb = code_weights[code][0]
            if p90 > p50:
                # Map reimbursement to 0-1 scale between median and 90th percentile
                pct = min(1.0, max(0.0, (avg_reimb - p50) / (p90 - p50)))
            else:
                pct = 0.5
            max_percentile = max(max_percentile, pct)
    
    return max_percentile


def compute_provider_code_frequency(provider_id, code, provider_code_counts):
    """Get the frequency of a specific code for a specific provider.
    
    Returns how often this provider bills this code (count).
    """
    key = (provider_id, code)
    return provider_code_counts.get(key, 0)


def compute_all_features(df, peer_stats, code_weights, provider_code_counts=None):
    """Compute all z-score features for every claim in the DataFrame.
    
    This is the main function called by both the ETL pipeline and the
    label injection step. It merges the pre-computed peer-group stats
    onto each claim row and computes the four z-score signals.
    
    Args:
        df: DataFrame with at least [claim_id, beneficiary_id, provider_id,
            provider_specialty, claim_type, reimbursement_amt, length_of_stay_days,
            diagnosis_codes, procedure_codes]
        peer_stats: DataFrame from compute_peer_group_stats()
        code_weights: dict from compute_code_reimbursement_weights()
        provider_code_counts: optional pre-computed dict of (provider_id, code) -> count
    
    Returns:
        DataFrame with additional feature columns:
        - reimbursement_zscore
        - los_zscore (NaN for non-inpatient)
        - visit_frequency_zscore
        - code_severity_percentile
        - high_severity_code_pct (provider-level)
    """
    logger.info(f"Computing features for {len(df)} claims...")
    
    # Build provider_code_counts if not provided
    if provider_code_counts is None:
        provider_code_counts = defaultdict(int)
        for _, row in df.iterrows():
            prov = row["provider_id"]
            for code in list(row.get("diagnosis_codes", [])) + list(row.get("procedure_codes", [])):
                if code and code.strip():
                    provider_code_counts[(prov, code)] += 1
    
    # Compute high-severity code percentage per provider
    # (fraction of a provider's codes that are in the top-decile by cost)
    if code_weights:
        all_avg_reimbs = np.array([v[0] for v in code_weights.values()])
        top_decile_threshold = np.percentile(all_avg_reimbs, 90) if len(all_avg_reimbs) > 0 else 0
    else:
        top_decile_threshold = 0
    
    high_sev_counts = defaultdict(lambda: [0, 0])  # [high_sev_count, total_count]
    for (prov, code), count in provider_code_counts.items():
        high_sev_counts[prov][1] += count
        if code in code_weights and code_weights[code][0] >= top_decile_threshold:
            high_sev_counts[prov][0] += count
    
    provider_high_sev_pct = {
        prov: (counts[0] / counts[1] if counts[1] > 0 else 0.0)
        for prov, counts in high_sev_counts.items()
    }
    
    # Merge peer-group stats onto claims
    merged = df.merge(
        peer_stats[
            ["provider_specialty", "claim_type", "provider_id",
             "avg_reimbursement", "reimbursement_stddev",
             "avg_length_of_stay", "los_stddev",
             "peer_avg_visit_freq", "peer_visit_freq_stddev"]
        ],
        on=["provider_specialty", "claim_type", "provider_id"],
        how="left"
    )
    
    # Compute z-scores
    # 1. Reimbursement z-score
    merged["reimbursement_zscore"] = (
        (merged["reimbursement_amt"] - merged["avg_reimbursement"])
        / merged["reimbursement_stddev"]
    ).fillna(0)
    
    # 2. LOS z-score (inpatient only)
    merged["los_zscore"] = np.where(
        merged["claim_type"] == "inpatient",
        (merged["length_of_stay_days"] - merged["avg_length_of_stay"])
        / merged["los_stddev"],
        np.nan
    )
    merged["los_zscore"] = merged["los_zscore"].fillna(0)
    
    # 3. Visit frequency z-score
    # For each claim, compute the beneficiary's YTD visit count to this provider
    visit_counts = (
        df.groupby(["beneficiary_id", "provider_id", "claim_year"])
        .size()
        .reset_index(name="bene_visits_ytd")
    )
    merged = merged.merge(
        visit_counts,
        on=["beneficiary_id", "provider_id", "claim_year"],
        how="left"
    )
    merged["bene_visits_ytd"] = merged["bene_visits_ytd"].fillna(1)
    
    merged["visit_frequency_zscore"] = (
        (merged["bene_visits_ytd"] - merged["peer_avg_visit_freq"])
        / merged["peer_visit_freq_stddev"]
    ).fillna(0)
    
    # 4. Code severity percentile (per claim)
    merged["code_severity_percentile"] = merged.apply(
        lambda row: compute_code_severity_percentile(
            row.get("diagnosis_codes", []),
            row.get("procedure_codes", []),
            code_weights
        ),
        axis=1
    )
    
    # Provider-level high severity code percentage
    merged["high_severity_code_pct"] = merged["provider_id"].map(
        provider_high_sev_pct
    ).fillna(0)
    
    # Code severity outlier: is the code in the provider's top-decile AND
    # is the code's frequency for this provider a peer-group outlier?
    merged["code_severity_outlier"] = (
        (merged["code_severity_percentile"] >= 0.9) &
        (merged["high_severity_code_pct"] > 0.3)  # provider uses high-sev codes more than peers
    ).astype(int)
    
    # Clean up temp columns
    merged = merged.drop(columns=["avg_reimbursement", "reimbursement_stddev",
                                   "avg_length_of_stay", "los_stddev",
                                   "peer_avg_visit_freq", "peer_visit_freq_stddev",
                                   "bene_visits_ytd"], errors="ignore")
    
    logger.info(f"Feature computation complete. {len(merged)} claims with features.")
    return merged
