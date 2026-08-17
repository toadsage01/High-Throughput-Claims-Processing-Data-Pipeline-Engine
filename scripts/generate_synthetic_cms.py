#!/usr/bin/env python3
"""
Synthetic CMS DE-SynPUF Data Generator
=====================================
Generates realistic synthetic beneficiary and claims data that follows
the DE-SynPUF schema. This is used because the actual CMS DE-SynPUF files
are not available in this environment.

Schema reference: https://www.cms.gov/Research-Statistics-Data-and-Systems/
  Downloadable-Public-Use-Files/SynPUFs/Downloads/SynPUF_DUG.pdf

Generated files:
  - data/cms/beneficiary.csv   (~10,000 beneficiaries)
  - data/cms/inpatient_claims.csv  (~5,000 claims)
  - data/cms/outpatient_claims.csv (~15,000 claims)
  - data/cms/carrier_claims.csv    (~35,000 claims)

Total: ~55,000 claim records across 10,000 beneficiaries.
"""

import csv
import random
import os
import string
from datetime import date, timedelta
from collections import defaultdict

random.seed(42)

OUTPUT_DIR = "data/cms"
N_BENEFICIARIES = 10_000
N_INPATIENT = 5_000
N_OUTPATIENT = 15_000
N_CARRIER = 35_000

# --- Specialty definitions ---
# In DE-SynPUF, provider specialty is typically found in carrier claims.
# We assign a specialty to each provider and carry it through all claim types.

SPECIALTIES = [
    "Internal Medicine", "Family Practice", "Cardiology", "Orthopedic Surgery",
    "General Surgery", "Neurology", "Oncology", "Radiology", "Anesthesiology",
    "Pathology", "Dermatology", "Urology", "Ophthalmology", "ENT",
    "Pulmonology", "Gastroenterology", "Psychiatry", "Physical Therapy",
    "Emergency Medicine", "General Practice"
]

SPECIALTY_WEIGHTS = [
    0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04, 0.03,
    0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03
]

# --- ICD-9 diagnosis codes (abbreviated realistic set) ---
DIAGNOSIS_CODES = [
    "25000", "25002", "25010", "25012",  # Diabetes mellitus
    "4011", "4019",                         # Essential hypertension
    "41401", "41400",                       # Coronary atherosclerosis
    "2720", "2721", "2722", "2724",       # Hyperlipidemia
    "496",                                  # COPD
    "5070",                                 # Pneumonia
    "53081",                                # GERD
    "5609",                                 # Intestinal obstruction
    "71590",                                # Osteoarthritis
    "7242",                                 # Lumbosacral radiculitis
    "78079",                                # Fatigue
    "78039",                                # Dizziness
    "78650",                                # Chest pain
    "78900",                                # Abdominal pain
    "79339",                                # Abnormal finding
    "V5861",                                # Long-term drug use
    "V5869",                                # Long-term drug use
    "E8788",                                # Surgical complications
    "4280",                                 # CHF
    "42731",                                # Atrial fibrillation
    "3429",                                 # Hemiplegia
    "5856",                                 # ESRD
    "0419",                                 # Bacterial infection
    "6826",                                 # Cellulitis
    "78609",                                # Respiratory distress
    "78097",                                # Altered mental status
]

# --- CPT/HCPCS procedure codes ---
PROCEDURE_CODES = [
    "99213", "99214", "99215", "99223", "99232", "99233",  # E/M codes
    "99281", "99282", "99283", "99284", "99285",            # ER visits
    "27447", "27130", "29881",                                 # Orthopedic procedures
    "33533", "33534",                                           # CABG
    "43239", "43235",                                           # Upper GI endoscopy
    "47562",                                                     # Lap cholecystectomy
    "71046", "74177", "72148",                                # Imaging
    "93000", "93010",                                           # ECG
    "93306",                                                     # Echo
    "70553",                                                     # Brain MRI
    "36561",                                                     # Central line
    "32557",                                                     # Thoracentesis
    "90834", "90837",                                           # Psychotherapy
    "97110", "97140",                                           # Physical therapy
    "20610",                                                     # Joint injection
    "36415",                                                     # Venipuncture
    "80053",                                                     # CMP
    "85025",                                                     # CBC
    "84443",                                                     # TSH
]

# --- Reimbursement distributions by claim type and specialty ---
# (mean, stddev) for generating realistic amounts
REIMB_PARAMS = {
    "inpatient": {
        "Cardiology": (18000, 8500), "General Surgery": (22000, 12000),
        "Orthopedic Surgery": (25000, 11000), "Pulmonology": (16000, 7000),
        "Neurology": (14000, 6000), "Oncology": (20000, 10000),
        "Emergency Medicine": (12000, 5000), "Internal Medicine": (11000, 5000),
        "General Practice": (9000, 4000), "Urology": (15000, 7000),
    },
    "outpatient": {
        "Cardiology": (3500, 2000), "Radiology": (2800, 1500),
        "Internal Medicine": (2200, 1200), "Family Practice": (1800, 900),
        "Oncology": (4500, 2500), "Gastroenterology": (3800, 2000),
        "Orthopedic Surgery": (4200, 2200), "Pulmonology": (2800, 1400),
        "General Practice": (1500, 800), "Dermatology": (1200, 600),
    },
    "carrier": {
        "Internal Medicine": (350, 200), "Family Practice": (280, 150),
        "Cardiology": (500, 300), "Radiology": (420, 250),
        "Dermatology": (220, 120), "Psychiatry": (380, 200),
        "Physical Therapy": (180, 80), "Ophthalmology": (450, 280),
        "General Practice": (200, 100), "Neurology": (520, 300),
    },
}

# Default params for specialty not in the dict
DEFAULT_REIMB = {"inpatient": (12000, 6000), "outpatient": (2500, 1300), "carrier": (300, 180)}

CHRONIC_CONDITIONS = [
    "Diabetes", "Hypertension", "HeartFailure", "IschemicHeart",
    "COPD", "Depression", "Arthritis", "KidneyDisease", "Cancer", "Stroke"
]


def rand_id(prefix, length=8):
    """Generate a random alphanumeric ID with a prefix."""
    chars = string.digits
    return prefix + ''.join(random.choice(chars) for _ in range(length))


def rand_date(start, end):
    """Generate a random date between start and end (inclusive)."""
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def generate_beneficiaries(n):
    """Generate beneficiary records following DE-SynPUF schema."""
    rows = []
    for i in range(n):
        bene_id = rand_id("BENE")
        birth_year = random.randint(1930, 1970)
        gender = random.choice(["M", "F"])
        state = random.choice([f"{j:02d}" for j in range(1, 51)])
        county = random.choice([f"{j:03d}" for j in range(1, 100)])
        # Chronic conditions (binary flags like DE-SynPUF SP_xxx columns)
        conditions = {}
        for cc in CHRONIC_CONDITIONS:
            conditions[cc] = 1 if random.random() < 0.15 else 0
        # Higher correlation for some
        if conditions["Diabetes"] == 1:
            conditions["Hypertension"] = 1 if random.random() < 0.6 else conditions["Hypertension"]
            conditions["KidneyDisease"] = 1 if random.random() < 0.2 else conditions["KidneyDisease"]
        if conditions["Hypertension"] == 1:
            conditions["HeartFailure"] = 1 if random.random() < 0.15 else conditions["HeartFailure"]
            conditions["IschemicHeart"] = 1 if random.random() < 0.2 else conditions["IschemicHeart"]

        # Medicare status
        part_a = random.choice([1, 1, 1, 0])  # 75% have Part A
        part_b = random.choice([1, 1, 1, 0])  # 75% have Part B

        row = {
            "DESYNPUF_ID": bene_id,
            "BENE_BIRTH_DT": f"{birth_year}0101",
            "BENE_SEX_IDENT_CD": str(gender),
            "BENE_STATE_CD": state,
            "BENE_COUNTY_CD": county,
            "BENE_RACE_CD": str(random.choice([1, 2, 3, 5])),
            "MEDICARE_ENTITLEMENT_CD": str(random.choice([1, 2, 3])),
            **{f"SP_{cc}": str(v) for cc, v in conditions.items()},
            "MEDREIMB_IP": str(round(random.gauss(5000, 3000), 2)),
            "MEDREIMB_OP": str(round(random.gauss(2000, 1200), 2)),
            "MEDREIMB_CAR": str(round(random.gauss(800, 500), 2)),
            "PLAN_CVRG_MOS_NUM": str(random.randint(1, 12)),
        }
        rows.append(row)
    return rows


def generate_providers(n_providers=500):
    """Generate a provider roster with assigned specialties."""
    providers = {}
    for i in range(n_providers):
        prov_id = rand_id("PRV")
        specialty = random.choices(SPECIALTIES, weights=SPECIALTY_WEIGHTS, k=1)[0]
        npi = rand_id("", 10)
        providers[prov_id] = {"specialty": specialty, "npi": npi}
    return providers


def generate_claims(claim_type, n, beneficiaries, providers, year=2009):
    """Generate claims records for a given claim type.
    
    Returns list of dicts with DE-SynPUF-style column names.
    For carrier claims, includes the PRVDR_SPCLTY column which is
    the provider specialty field in the real data.
    """
    bene_ids = [b["DESYNPUF_ID"] for b in beneficiaries]
    prov_ids = list(providers.keys())
    rows = []
    
    claim_prefix = {"inpatient": "IPA", "outpatient": "OPA", "carrier": "CAR"}
    
    # LOS distribution by claim type
    if claim_type == "inpatient":
        los_mean, los_std = 5.5, 4.0
    else:
        los_mean, los_std = 0, 0
    
    for i in range(n):
        claim_id = rand_id(claim_prefix[claim_type])
        bene_id = random.choice(bene_ids)
        prov_id = random.choice(prov_ids)
        specialty = providers[prov_id]["specialty"]
        
        # Claim dates
        start_date = rand_date(date(year, 1, 1), date(year, 12, 31))
        if claim_type == "inpatient":
            los = max(1, int(random.gauss(los_mean, los_std)))
            end_date = start_date + timedelta(days=los)
        else:
            los = 0
            end_date = start_date + timedelta(days=random.randint(0, 1))
        
        # Diagnosis codes (1-10 per claim, first is principal)
        n_dx = random.randint(1, min(5, 10))
        dx_codes = random.sample(DIAGNOSIS_CODES, min(n_dx, len(DIAGNOSIS_CODES)))
        # Pad to 10 positions (ICD9_DGNS_CD1..ICD9_DGNS_CD10 in DE-SynPUF)
        dx_padded = dx_codes + [""] * (10 - len(dx_codes))
        
        # Procedure codes (1-6 for inpatient/outpatient, 1-3 for carrier)
        max_proc = 6 if claim_type != "carrier" else 3
        n_proc = random.randint(0, min(random.randint(1, 3), max_proc))
        proc_codes = random.sample(PROCEDURE_CODES, min(n_proc, len(PROCEDURE_CODES)))
        proc_padded = proc_codes + [""] * (max_proc - len(proc_codes))
        
        # Reimbursement amount — based on specialty + claim type
        params = REIMB_PARAMS.get(claim_type, {}).get(specialty) or DEFAULT_REIMB[claim_type]
        # 5% of claims are outliers (high cost)
        if random.random() < 0.05:
            reimb = random.gauss(params[0] * 2.5, params[1] * 2.0)
        else:
            reimb = max(0, random.gauss(params[0], params[1]))
        reimb = round(reimb, 2)
        
        # Build the row — DE-SynPUF column naming conventions
        row = {
            "CLM_ID": claim_id,
            "DESYNPUF_ID": bene_id,
            "PRVDR_NUM": prov_id,
            "CLM_FROM_DT": start_date.strftime("%Y%m%d"),
            "CLM_THRU_DT": end_date.strftime("%Y%m%d"),
            "CLM_ADMTG_DT": start_date.strftime("%Y%m%d") if claim_type == "inpatient" else "",
            "NCH_BENE_DSCHRG_DT": end_date.strftime("%Y%m%d") if claim_type == "inpatient" else "",
        }
        
        # Diagnosis code columns
        for idx, code in enumerate(dx_padded):
            col = f"ICD9_DGNS_CD{idx+1}" if claim_type != "carrier" else f"ICD9_DGNS_CD{idx+1}"
            row[col] = code
        
        # Procedure code columns
        proc_col_prefix = "HCPCS_CD" if claim_type == "carrier" else "ICD9_PRCDR_CD"
        for idx, code in enumerate(proc_padded):
            row[f"{proc_col_prefix}{idx+1}"] = code
        
        # Reimbursement columns (DE-SynPUF naming)
        if claim_type == "carrier":
            row["CARR_CLM_MTCHD_CNTRY_USRD_CD"] = "1"
            row["CARR_CLM_PRMRY_PYR_CD"] = str(random.randint(0, 3))
            row["CARR_LINE_PRVDR_TYPE_CD"] = str(random.randint(1, 85))
            # CRITICAL: PRVDR_SPCLTY is the provider specialty field in carrier claims
            # This is the column we use for peer-group assignment
            row["PRVDR_SPCLTY"] = specialty
            row["LINE_NCH_PMT_AMT"] = str(reimb)
            row["LINE_CARR_CLM_SBMTD_CHRG_AMT"] = str(round(reimb * random.uniform(1.2, 2.0), 2))
            row["NCH_CARR_CLM_ALOWD_AMT"] = str(round(reimb * random.uniform(0.9, 1.1), 2))
        else:
            row["CLM_PMT_AMT"] = str(reimb)
            row["CLM_TOT_CHRG_AMT"] = str(round(reimb * random.uniform(1.3, 2.5), 2))
            row["NCH_PRMRY_PYR_CLM_PD_AMT"] = str(round(reimb * random.uniform(0, 0.3), 2))
            if claim_type == "inpatient":
                # Inpatient also has CLM_UTLZTN_DT and facility columns
                row["CLM_UTLZTN_CD"] = str(random.randint(1, 3))
        
        # Store LOS for inpatient
        row["_LOS"] = str(los)
        row["_CLAIM_TYPE"] = claim_type
        row["_SPECIALTY"] = specialty
        
        rows.append(row)
    
    return rows


def write_csv(filepath, rows, fieldnames):
    """Write rows to CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows to {filepath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating synthetic CMS DE-SynPUF data...")
    print(f"  Beneficiaries: {N_BENEFICIARIES:,}")
    print(f"  Inpatient claims: {N_INPATIENT:,}")
    print(f"  Outpatient claims: {N_OUTPATIENT:,}")
    print(f"  Carrier claims: {N_CARRIER:,}")
    print(f"  Total claims: {N_INPATIENT + N_OUTPATIENT + N_CARRIER:,}")
    print()
    
    # Generate providers
    providers = generate_providers(500)
    
    # Generate beneficiaries
    print("[1/4] Generating beneficiaries...")
    beneficiaries = generate_beneficiaries(N_BENEFICIARIES)
    bene_fields = list(beneficiaries[0].keys())
    write_csv(os.path.join(OUTPUT_DIR, "beneficiary.csv"), beneficiaries, bene_fields)
    
    # Generate inpatient claims
    print("[2/4] Generating inpatient claims...")
    inpatient = generate_claims("inpatient", N_INPATIENT, beneficiaries, providers)
    ip_fields = list(inpatient[0].keys())
    write_csv(os.path.join(OUTPUT_DIR, "inpatient_claims.csv"), inpatient, ip_fields)
    
    # Generate outpatient claims
    print("[3/4] Generating outpatient claims...")
    outpatient = generate_claims("outpatient", N_OUTPATIENT, beneficiaries, providers)
    op_fields = list(outpatient[0].keys())
    write_csv(os.path.join(OUTPUT_DIR, "outpatient_claims.csv"), outpatient, op_fields)
    
    # Generate carrier claims
    print("[4/4] Generating carrier claims...")
    carrier = generate_claims("carrier", N_CARRIER, beneficiaries, providers)
    car_fields = list(carrier[0].keys())
    write_csv(os.path.join(OUTPUT_DIR, "carrier_claims.csv"), carrier, car_fields)
    
    print()
    print("Done! Synthetic data written to:", OUTPUT_DIR)
    print()
    print("IMPORTANT: Provider specialty field confirmation")
    print("  In carrier claims, the specialty is stored in column: PRVDR_SPCLTY")
    print("  This is the column used for peer-group assignment.")
    print("  For inpatient/outpatient claims, specialty is derived from the")
    print("  provider roster (PRVDR_NUM -> specialty lookup).")


if __name__ == "__main__":
    main()
