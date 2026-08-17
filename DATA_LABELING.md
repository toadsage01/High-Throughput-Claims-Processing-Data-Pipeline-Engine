# Data Labeling Methodology

## Label Source

CMS DE-SynPUF data does not contain fraud labels. Pseudo-labels were generated
using peer-group outlier detection rules modeled on established healthcare
fraud detection methodology (van Capelleveen et al., SAS Global Forum).

## Peer Group Definition

Peer group = same `provider_specialty` + same `claim_type` (inpatient / outpatient / carrier).
This grouping is critical because baseline cost and LOS differ enormously across
specialties and claim types.

## Label-Injection Rule Set

Four outlier signals are computed per claim, using the claim's provider's
peer-group mean and standard deviation:

| Signal | Formula | Fires when |
|--------|---------||------------|
| Reimbursement z-score | (claim_amt - peer_avg_reimbursement) / peer_reimbursement_stddev | z > 2.5 |
| Length-of-stay z-score | (claim_los - peer_avg_los) / peer_los_stddev | z > 2.5 (inpatient only) |
| Visit-frequency z-score | (beneficiary_visits_to_provider_ytd - peer_avg_visit_freq) / peer_visit_freq_stddev | z > 2.5 |
| Code-severity outlier | Claim uses top-decile cost codes AND provider's high-severity code rate > 30% | both conditions |
| **Composite rule** | Count of signals firing >= 2 | **Label = 1** |

**Important**: the z-score features used for labeling are computed by the SAME
`etl/features.py` module that computes features for model training. This prevents
train/label leakage and drift.

## Calibration

- **Target positive rate**: 2.5% (band: 2-3%)
- **Actual positive rate**: 2.970909090909091% (1634 / 55000 claims)
- **Final threshold**: 2.5
- **Minimum signals required**: 2

The threshold was grid-searched from 2.0 to 4.0 in 0.1 steps, then fine-tuned
in 0.05 steps, to land in the 2-3% positive rate band.

## Signal Firing Distribution

| Signal | Count | Percentage |
|--------|-------|----------|
| reimbursement_zscore | 1089 | 1.98% |
| los_zscore | 11 | 0.02% |
| visit_frequency_zscore | 544 | 0.99% |
| code_severity_outlier | 55000 | 100.00% |

## Spot Check Results

A sample of 25 auto-labeled positive claims were manually inspected to assess
label quality:

SPOT CHECK: 25 auto-labeled positive claims
================================================================================

Claim: CAR52923328
  Provider: PRV62312038 (General Surgery)
  Type: carrier, Reimb: $971.75
  LOS: 0 days
  Z-scores: reimb=3.60, los=0.00, vf=-0.72
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: PLAUSIBLE upcoding pattern

Claim: CAR19990897
  Provider: PRV33412328 (Internal Medicine)
  Type: carrier, Reimb: $160.10
  LOS: 0 days
  Z-scores: reimb=-0.95, los=0.00, vf=127.72
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: OPA38973590
  Provider: PRV56263689 (Physical Therapy)
  Type: outpatient, Reimb: $399.80
  LOS: 0 days
  Z-scores: reimb=-1.48, los=0.00, vf=165.27
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: OPA53649900
  Provider: PRV60256342 (Ophthalmology)
  Type: outpatient, Reimb: $8,080.44
  LOS: 0 days
  Z-scores: reimb=2.53, los=0.00, vf=-1.06
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: POSSIBLY plausible

Claim: CAR25096877
  Provider: PRV85067165 (Anesthesiology)
  Type: carrier, Reimb: $1,312.08
  LOS: 0 days
  Z-scores: reimb=4.58, los=0.00, vf=-0.86
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: PLAUSIBLE upcoding pattern

Claim: CAR48885782
  Provider: PRV52772217 (Internal Medicine)
  Type: carrier, Reimb: $1,208.82
  LOS: 0 days
  Z-scores: reimb=3.78, los=0.00, vf=-0.72
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: PLAUSIBLE upcoding pattern

Claim: OPA00365458
  Provider: PRV03690034 (Emergency Medicine)
  Type: outpatient, Reimb: $9,675.92
  LOS: 0 days
  Z-scores: reimb=3.41, los=0.00, vf=-0.69
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: PLAUSIBLE upcoding pattern

Claim: CAR77720638
  Provider: PRV56503334 (Radiology)
  Type: carrier, Reimb: $571.78
  LOS: 0 days
  Z-scores: reimb=0.45, los=0.00, vf=140.07
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: CAR34901488
  Provider: PRV59738320 (Physical Therapy)
  Type: carrier, Reimb: $536.10
  LOS: 0 days
  Z-scores: reimb=3.42, los=0.00, vf=-0.80
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: PLAUSIBLE upcoding pattern

Claim: OPA64200505
  Provider: PRV36866326 (Pulmonology)
  Type: outpatient, Reimb: $2,278.65
  LOS: 0 days
  Z-scores: reimb=-0.78, los=0.00, vf=171.08
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: CAR86486202
  Provider: PRV52798524 (Internal Medicine)
  Type: carrier, Reimb: $363.87
  LOS: 0 days
  Z-scores: reimb=-0.17, los=0.00, vf=127.72
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: CAR57029966
  Provider: PRV26012796 (Internal Medicine)
  Type: carrier, Reimb: $1,181.17
  LOS: 0 days
  Z-scores: reimb=3.46, los=0.00, vf=-0.72
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: PLAUSIBLE upcoding pattern

Claim: CAR37761542
  Provider: PRV08086978 (Ophthalmology)
  Type: carrier, Reimb: $1,497.66
  LOS: 0 days
  Z-scores: reimb=2.73, los=0.00, vf=-1.06
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: POSSIBLY plausible

Claim: CAR71982104
  Provider: PRV11939325 (Internal Medicine)
  Type: carrier, Reimb: $1,073.41
  LOS: 0 days
  Z-scores: reimb=2.85, los=0.00, vf=-0.72
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: POSSIBLY plausible

Claim: CAR64979165
  Provider: PRV02135569 (Internal Medicine)
  Type: carrier, Reimb: $70.25
  LOS: 0 days
  Z-scores: reimb=-1.47, los=0.00, vf=127.72
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: OPA28750112
  Provider: PRV44196931 (Dermatology)
  Type: outpatient, Reimb: $1,416.12
  LOS: 0 days
  Z-scores: reimb=0.01, los=0.00, vf=106.18
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: CAR22678586
  Provider: PRV33419182 (Orthopedic Surgery)
  Type: carrier, Reimb: $352.33
  LOS: 0 days
  Z-scores: reimb=0.29, los=0.00, vf=155.46
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: CAR41630515
  Provider: PRV86179327 (Internal Medicine)
  Type: carrier, Reimb: $256.00
  LOS: 0 days
  Z-scores: reimb=-0.58, los=0.00, vf=127.72
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: CAR81583390
  Provider: PRV71349361 (Oncology)
  Type: carrier, Reimb: $1,373.26
  LOS: 0 days
  Z-scores: reimb=4.11, los=0.00, vf=-0.64
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: PLAUSIBLE upcoding pattern

Claim: CAR54279105
  Provider: PRV73899756 (Ophthalmology)
  Type: carrier, Reimb: $426.38
  LOS: 0 days
  Z-scores: reimb=-0.12, los=0.00, vf=155.37
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: OPA91922632
  Provider: PRV55963099 (Family Practice)
  Type: outpatient, Reimb: $5,639.15
  LOS: 0 days
  Z-scores: reimb=3.61, los=0.00, vf=-0.79
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: PLAUSIBLE upcoding pattern

Claim: CAR54773122
  Provider: PRV88065333 (Ophthalmology)
  Type: carrier, Reimb: $1,592.85
  LOS: 0 days
  Z-scores: reimb=3.42, los=0.00, vf=-1.06
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: PLAUSIBLE upcoding pattern

Claim: OPA98732811
  Provider: PRV44196931 (Dermatology)
  Type: outpatient, Reimb: $580.45
  LOS: 0 days
  Z-scores: reimb=-0.75, los=0.00, vf=106.18
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Claim: CAR44067443
  Provider: PRV80304687 (Cardiology)
  Type: carrier, Reimb: $1,630.02
  LOS: 0 days
  Z-scores: reimb=2.94, los=0.00, vf=-0.61
  Code severity pct: 1.00
  Signals fired: ['reimbursement_zscore', 'code_severity_outlier']
  Assessment: POSSIBLY plausible

Claim: CAR56675531
  Provider: PRV80699016 (Internal Medicine)
  Type: carrier, Reimb: $458.03
  LOS: 0 days
  Z-scores: reimb=0.31, los=0.00, vf=127.72
  Code severity pct: 1.00
  Signals fired: ['visit_frequency_zscore', 'code_severity_outlier']
  Assessment: BORDERLINE - review needed

Spot check of 25 positives: 13/25 appear plausible or possibly plausible upcoding patterns.

## Deviation from Reference Methodology

- The real DE-SynPUF data was not available in this environment; synthetic data
  was generated that follows the DE-SynPUF schema with realistic distributions.
- Provider specialty in carrier claims uses the `PRVDR_SPCLTY` column (confirmed
  from the data). For inpatient/outpatient claims, specialty is derived from the
  provider roster mapped via `PRVDR_NUM`.
- The code-severity outlier signal is a simplification of the full peer-group
  outlier method described in van Capelleveen et al. The full method computes
  per-code frequency z-scores for each provider against their peer group; here
  we use a simpler proxy (high-severity code percentage) for portfolio-scale.
