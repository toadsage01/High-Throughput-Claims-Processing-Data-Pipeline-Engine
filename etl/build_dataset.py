"""
PySpark ETL Job - Build Feature-Engineered Dataset
=======================================================
Reads raw CMS DE-SynPUF CSVs, joins beneficiary + claims files, normalizes
diagnosis/procedure codes (ICD-9/CPT/HCPCS), and computes provider
peer-group statistics and z-score features using the shared etl/features.py module.

Output:
  - data/processed/claims_featured.parquet  — cleaned, feature-engineered claims
  - data/processed/provider_peer_stats.parquet  — peer-group stats per provider
  - data/processed/beneficiary_profile.parquet — beneficiary summaries

Usage:
    python3 etl/build_dataset.py --cms-dir data/cms --output-dir data/processed

Note: Runs PySpark in local mode. For larger datasets, submit to a proper
  Spark cluster with --master yarn or --master spark://...
"""

import argparse
import logging
import os
import sys
import numpy as np

import pandas as pd
from collections import defaultdict

# We use PySpark for the heavy ETL transforms. For the feature computation
# (which requires per-provider peer-group stats), we drop to Pandas since
# the shared features.py module is Pandas-based and the dataset is manageable.
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, DateType,
    ArrayType, FloatType
)

from etl.features import (
    compute_peer_group_stats,
    compute_code_reimbursement_weights,
    compute_all_features
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# CMS DE-SynPUF column mappings
SPECIALTY_COLUMN = "PRVDR_SPCLTY"


def parse_date(date_str):
    """Parse YYYYMMDD or YYYYMM string to YYYY-MM-DD."""
    if not date_str or pd.isna(date_str) or str(date_str).strip() == "":
        return None
    s = str(date_str).strip()
    if len(s) >= 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    elif len(s) >= 6:
        return f"{s[:4]}-{s[4:6]}-01"
    return None


def extract_diagnosis_codes(row, n_cols=10, prefix="ICD9_DGNS_CD"):
    """Extract non-empty diagnosis codes from numbered columns."""
    codes = []
    for i in range(1, n_cols + 1):
        col = f"{prefix}{i}"
        val = row.get(col, "")
        if val and str(val).strip():
            codes.append(str(val).strip())
    return codes


def extract_procedure_codes(row, claim_type, n_cols=6):
    """Extract non-empty procedure codes from numbered columns."""
    prefix = "HCPCS_CD" if claim_type == "carrier" else "ICD9_PRCDR_CD"
    codes = []
    for i in range(1, n_cols + 1):
        col = f"{prefix}{i}"
        val = row.get(col, "")
        if val and str(val).strip():
            codes.append(str(val).strip())
    return codes


def read_and_normalize_claims(spark, cms_dir):
    """Read all three claim types from CSV, normalize to a common schema."""
    logger.info("Reading raw CMS CSV files...")

    all_claims_pd = []

    # Read beneficiary data
    bene_path = os.path.join(cms_dir, "beneficiary.csv")
    if not os.path.exists(bene_path):
        raise FileNotFoundError(f"Beneficiary file not found: {bene_path}")

    bene_df = pd.read_csv(bene_path, dtype=str)
    logger.info(f"  Beneficiaries loaded: {len(bene_df)}")

    # Build provider -> specialty mapping from carrier claims
    carrier_path = os.path.join(cms_dir, "carrier_claims.csv")
    provider_specialty_map = {}
    if os.path.exists(carrier_path):
        carrier_raw = pd.read_csv(carrier_path, dtype=str)
        provider_specialty_map = (
            carrier_raw[["PRVDR_NUM", SPECIALTY_COLUMN]]
            .drop_duplicates(subset="PRVDR_NUM")
            .set_index("PRVDR_NUM")[SPECIALTY_COLUMN]
            .to_dict()
        )
        logger.info(f"  Provider specialty map: {len(provider_specialty_map)} providers")

    # Process each claim type
    claim_configs = [
        ("inpatient_claims.csv", "inpatient", "CLM_PMT_AMT"),
        ("outpatient_claims.csv", "outpatient", "CLM_PMT_AMT"),
        ("carrier_claims.csv", "carrier", "LINE_NCH_PMT_AMT"),
    ]

    for filename, claim_type, reimb_col in claim_configs:
        filepath = os.path.join(cms_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"  {filename} not found, skipping")
            continue

        raw = pd.read_csv(filepath, dtype=str)
        logger.info(f"  {filename}: {len(raw)} rows")

        n_dropped = 0
        n_malformed = 0
        processed = []

        for _, row in raw.iterrows():
            try:
                claim_id = str(row.get("CLM_ID", "")).strip()
                if not claim_id:
                    n_dropped += 1
                    continue

                bene_id = str(row.get("DESYNPUF_ID", "")).strip()
                prov_id = str(row.get("PRVDR_NUM", "")).strip()
                if not bene_id or not prov_id:
                    n_dropped += 1
                    continue

                # Parse dates
                start_dt = parse_date(row.get("CLM_FROM_DT", ""))
                end_dt = parse_date(row.get("CLM_THRU_DT", ""))
                if not start_dt:
                    n_malformed += 1
                    continue

                # Reimbursement amount
                try:
                    reimb = float(row.get(reimb_col, 0))
                except (ValueError, TypeError):
                    reimb = 0.0

                # LOS (only for inpatient)
                if claim_type == "inpatient":
                    los_str = row.get("_LOS", "0")
                    try:
                        los = int(float(los_str))
                    except (ValueError, TypeError):
                        if end_dt and start_dt:
                            los = max(0, (pd.Timestamp(end_dt) - pd.Timestamp(start_dt)).days)
                        else:
                            los = 0
                else:
                    los = 0

                # Provider specialty
                specialty = row.get(SPECIALTY_COLUMN, "")
                if not specialty or str(specialty).strip() == "":
                    specialty = provider_specialty_map.get(prov_id, "Unknown")
                specialty = str(specialty).strip()

                # Diagnosis codes
                diag_codes = extract_diagnosis_codes(row, n_cols=10)

                # Procedure codes
                n_proc_cols = 6 if claim_type != "carrier" else 3
                proc_codes = extract_procedure_codes(row, claim_type, n_cols=n_proc_cols)

                # Principal diagnosis
                admit_dx = diag_codes[0] if diag_codes else ""

                # Year from start date
                claim_year = int(start_dt[:4]) if start_dt else 2009

                processed.append({
                    "claim_id": claim_id,
                    "beneficiary_id": bene_id,
                    "provider_id": prov_id,
                    "provider_specialty": specialty,
                    "claim_type": claim_type,
                    "claim_start_date": start_dt,
                    "claim_end_date": end_dt,
                    "claim_year": claim_year,
                    "admit_diagnosis_cd": admit_dx,
                    "diagnosis_codes": diag_codes,
                    "procedure_codes": proc_codes,
                    "reimbursement_amt": reimb,
                    "length_of_stay_days": los,
                })

            except Exception as e:
                n_malformed += 1
                continue

        claims_pd = pd.DataFrame(processed)
        all_claims_pd.append(claims_pd)
        logger.info(f"  {claim_type}: {len(claims_pd)} valid claims "
                    f"(dropped {n_dropped} null, {n_malformed} malformed)")

    combined = pd.concat(all_claims_pd, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} claims")

    return combined, bene_df, provider_specialty_map


def build_beneficiary_profiles(bene_df, claims_df):
    """Build beneficiary profile summary."""
    logger.info("Building beneficiary profiles...")

    chronic_cols = [c for c in bene_df.columns if c.startswith("SP_")]

    profiles = []
    for _, row in bene_df.iterrows():
        conditions = set()
        for col in chronic_cols:
            if str(row.get(col, "0")) == "1":
                cond_name = col.replace("SP_", "")
                conditions.add(cond_name)

        birth_str = str(row.get("BENE_BIRTH_DT", "19400101"))
        birth_year = int(birth_str[:4]) if len(birth_str) >= 4 else 1940

        profiles.append({
            "beneficiary_id": str(row["DESYNPUF_ID"]),
            "birth_year": birth_year,
            "chronic_conditions": list(conditions),
        })

    profiles_df = pd.DataFrame(profiles)

    # Add YTD claim count
    claim_counts = (
        claims_df.groupby(["beneficiary_id", "claim_year"])
        .size()
        .reset_index(name="claim_count")
    )
    latest_counts = claim_counts.loc[
        claim_counts.groupby("beneficiary_id")["claim_year"].idxmax()
    ]
    latest_counts = latest_counts[["beneficiary_id", "claim_count"]].rename(
        columns={"claim_count": "claim_count_ytd"}
    )

    profiles_df = profiles_df.merge(latest_counts, on="beneficiary_id", how="left")
    profiles_df["claim_count_ytd"] = profiles_df["claim_count_ytd"].fillna(0).astype(int)

    logger.info(f"  Built {len(profiles_df)} beneficiary profiles")
    return profiles_df


def main():
    parser = argparse.ArgumentParser(description="ETL: Build feature-engineered claims dataset")
    parser.add_argument("--cms-dir", default="data/cms", help="Directory with raw CMS CSVs")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Spark (local mode)
    logger.info("Initializing PySpark (local mode)...")
    spark = SparkSession.builder \
        .appName("ClaimsETL") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        # Step 1: Read and normalize claims
        logger.info("=== STEP 1: Reading and normalizing raw CMS data ===")
        claims_df, bene_df, _ = read_and_normalize_claims(spark, args.cms_dir)

        logger.info(f"Claims per type:\n{claims_df['claim_type'].value_counts().to_string()}")
        logger.info(f"Specialties (top 10):\n"
                    f"{claims_df['provider_specialty'].value_counts().head(10).to_string()}")
        logger.info(f"Reimbursement stats:\n{claims_df['reimbursement_amt'].describe().to_string()}")

        # Step 2: Compute provider peer-group statistics
        logger.info("=== STEP 2: Computing peer-group statistics ===")
        peer_stats = compute_peer_group_stats(claims_df)

        # Step 3: Compute code reimbursement weights
        logger.info("=== STEP 3: Computing code reimbursement weights ===")
        code_weights = compute_code_reimbursement_weights(claims_df)

        # Step 4: Compute all features for every claim
        logger.info("=== STEP 4: Computing z-score features ===")
        featured_claims = compute_all_features(claims_df, peer_stats, code_weights)

        # Step 5: Build beneficiary profiles
        logger.info("=== STEP 5: Building beneficiary profiles ===")
        bene_profiles = build_beneficiary_profiles(bene_df, claims_df)

        # Step 6: Save outputs
        logger.info("=== STEP 6: Saving outputs ===")

        claims_output = os.path.join(args.output_dir, "claims_featured.parquet")
        featured_claims.to_parquet(claims_output, index=False)
        logger.info(f"  Saved {len(featured_claims)} claims to {claims_output}")

        peer_output = os.path.join(args.output_dir, "provider_peer_stats.parquet")
        peer_stats.to_parquet(peer_output, index=False)
        logger.info(f"  Saved {len(peer_stats)} provider peer stats to {peer_output}")

        bene_output = os.path.join(args.output_dir, "beneficiary_profile.parquet")
        bene_profiles.to_parquet(bene_output, index=False)
        logger.info(f"  Saved {len(bene_profiles)} beneficiary profiles to {bene_output}")

        # Summary
        logger.info("=== ETL COMPLETE ===")
        logger.info(f"Total claims: {len(featured_claims)}")
        logger.info(f"Columns: {list(featured_claims.columns)}")

        for col in ["reimbursement_zscore", "los_zscore", "visit_frequency_zscore",
                     "code_severity_percentile"]:
            if col in featured_claims.columns:
                logger.info(f"  {col}: mean={featured_claims[col].mean():.4f}, "
                            f"std={featured_claims[col].std():.4f}, "
                            f"p95={featured_claims[col].quantile(0.95):.4f}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
