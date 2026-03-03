# UOB 2-Hour Simulated Intelligent Automation Coding Assessment

## Candidate Brief

You are a Data Scientist supporting UOB Operations Intelligence. Build a compact analysis from messy operational data and present actionable recommendations.

- **Time limit:** 120 minutes
- **Language:** Python
- **Core tools:** pandas, numpy, scikit-learn (optional for Task 4)
- **Dataset:** `uob_2hr_assessment_dataset.csv`

## Suggested Time Allocation

- **Task 1 (Cleaning):** 25 min
- **Task 2 (KPI):** 30 min
- **Task 3 (Insights):** 25 min
- **Task 4 (Automation/ML):** 30 min
- **Wrap-up summary:** 10 min

---

## Dataset Scope

The dataset combines four banking operations streams:

- Contact center interactions
- ATM cash status
- Workforce staffing snapshots
- Document processing quality

Total raw rows: **27**

---

## Task 1 — Data Cleaning (Foundational)

Clean and standardize the dataset for downstream analytics.

### Requirements

1. Parse timestamps robustly (`errors='coerce'`).
2. Remove duplicate events by `event_id`, keeping the latest record.
3. Convert numeric-like fields to numeric (`queue_time_sec`, `handle_time_sec`, `atm_capacity`, `atm_withdrawn`, `ocr_confidence`, `doc_errors`, `planned_staff`, `actual_staff`).
4. Handle malformed numerics (e.g., comma separators, `%`, `N/A`).
5. Replace missing `agent_id` with `UNASSIGNED`.
6. Treat negative handle times as invalid (`NaN`), then impute if needed.

### Expected output checks (reference)

- Raw rows: **27**
- Rows after dedup on `event_id`: **26**
- Invalid timestamps after parsing: **1**
- Contact-center row with missing `agent_id` mapped to `UNASSIGNED`: **1**
- One negative `handle_time_sec` detected and corrected

---

## Task 2 — KPI Computation (Intermediate)

Build KPI tables for contact center and ATM.

### Requirements

1. For contact center rows (valid timestamps only), compute by `team`:
   - `total_calls`
   - `abandon_rate`
   - `sla_breach_rate` where `queue_time_sec > 120`
   - `avg_queue_time_sec`
2. For ATM rows, compute:
   - `atm_utilization = atm_withdrawn / atm_capacity`
   - Top-up risk flag (`atm_utilization > 0.85`)
3. For workforce rows, compute:
   - `staff_gap = planned_staff - actual_staff`

### Expected KPI output (reference, after Task 1 rules)

Contact center team KPI:

- **Tier1** → total_calls: **5**, abandon_rate: **0.20**, sla_breach_rate: **0.20**, avg_queue_time_sec: **81.60**
- **Tier2** → total_calls: **4**, abandon_rate: **0.25**, sla_breach_rate: **0.50**, avg_queue_time_sec: **126.25**
- **Tier3** → total_calls: **4**, abandon_rate: **0.25**, sla_breach_rate: **0.50**, avg_queue_time_sec: **465.00**

ATM utilization highlights:

- **ATM01**: 0.860 (high risk)
- **ATM04**: 1.006 (high risk, suspicious >100% utilization)
- **ATM03**: capacity is 0 (must guard divide-by-zero)

---

## Task 3 — Insight Generation (Advanced Analytics)

Generate concise operational insights for leadership.

### Requirements

1. Identify the top operational pain points (max 3) across contact center, ATM, workforce, and document processing.
2. Quantify each insight with KPI evidence.
3. Provide one recommendation per insight with likely impact.

### Expected high-quality insights (example)

1. **Tier3 contact center risk concentration:** same SLA breach rate as Tier2 (0.50) but far higher average queue time (465s), indicating severe outlier pressure.
2. **ATM cash service risk:** ATM01 and ATM04 exceed 85% utilization, with ATM04 >100%, suggesting timing mismatch or data integrity issue.
3. **Document-processing quality risk:** low OCR confidence and high document errors cluster in Claims/Trade rows, increasing manual rework probability.

---

## Task 4 — Automation Logic or ML Prediction (Stretch)

Pick **one** path.

### Option A: Rule-based automation

Design operational triggers and produce an action list.

Required rule set:

1. Trigger callback queue if `queue_time_sec > 120` and `abandoned == 1`.
2. Trigger ATM top-up alert if `atm_utilization > 0.85`.
3. Trigger document manual-review queue if `ocr_confidence < 0.8` or `doc_errors >= 3`.

Expected flagged records (reference):

- Callback candidates: **CC002, CC005, CC009**
- ATM top-up candidates: **ATM01, ATM04**
- Document manual-review candidates: **DOC002, DOC004**

### Option B: Basic ML baseline

Build a binary classifier for `risk_label` (or contact-center `sla_breach`).

Required output:

1. Feature selection and preprocessing summary.
2. Train/test split.
3. Logistic Regression and Random Forest baseline comparison.
4. Precision/recall interpretation and threshold recommendation.

Expected quality bar:

- Candidate explains class imbalance risk and avoids overclaiming on small sample size.
- Candidate proposes human-in-the-loop deployment.

---

## Hidden Edge Cases in Dataset (for interviewer)

Use these to evaluate robustness (do not reveal to candidate initially):

1. Duplicate `event_id` (`CC007`) with updated status.
2. One malformed timestamp (`bad_timestamp`).
3. Mixed timestamp format (`03/02/2026 11:40`).
4. Numeric string with comma (`"1,200"`).
5. Numeric placeholder (`N/A`).
6. Percentage string in numeric column (`ocr_confidence = "91%"`).
7. Negative handle time (`-20`).
8. Missing `agent_id`.
9. ATM row with zero capacity (division-by-zero risk).
10. ATM utilization above 100% (data integrity alert).

---

## Interviewer Follow-up Questions

1. Why did you choose your deduplication rule (`keep='last'`) and what are alternatives?
2. How would your KPI logic change if timestamps are in different time zones?
3. If Tier2 and Tier3 both have 50% SLA breach, why prioritize one over the other?
4. How would you productionize the rule engine (scheduling, monitoring, audit trail)?
5. What false-positive risks exist in your anomaly/automation logic?
6. For ML: how do you prevent leakage and evaluate reliability on small data?
7. How would you communicate uncertainty to operations leadership?

---

## Model High-Scoring Solution Outline

1. **Data engineering quality (30%)**
   - Robust parsing/coercion, safe handling of malformed values, explicit assumptions.
2. **Analytical correctness (30%)**
   - Accurate KPI formulas, sound grouping logic, clear validation checks.
3. **Operational insight (20%)**
   - Prioritized findings with quantified impact and practical interventions.
4. **Automation/ML implementation (15%)**
   - Working rule logic or baseline model with honest limitations.
5. **Communication (5%)**
   - Clear, concise executive summary (problem → evidence → action).

### High-scoring narrative pattern

- “We cleaned and standardized 27 rows down to 26 valid records after deduplication, then identified Tier3 queue concentration and ATM over-utilization as immediate service risks. Recommended callback and top-up triggers are directly traceable to KPI thresholds and can be operationalized with daily batch monitoring plus analyst review.”

---

## Deliverables Candidate Should Submit

1. One reproducible Python script or notebook.
2. Cleaned dataset preview and data-quality checks.
3. KPI tables and top risk findings.
4. Rule/ML output table with flagged records.
5. 5-bullet business recommendation summary.
