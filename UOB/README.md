# UOB Intelligent Automation Technical Assessment Guide

Professional revision pack for a 2-hour on-site coding test focused on:

- Python + Pandas + NumPy
- Data cleaning and KPI analytics
- Automation logic and anomaly detection
- Basic ML (Logistic Regression, Random Forest)
- Business-impact communication

## Assessment Format (what to expect)

- **Duration:** 2 hours (coding + explanation)
- **Typical flow:**
  1. Load and clean operational datasets
  2. Calculate KPIs and identify risk/performance issues
  3. Implement rule-based automation logic
  4. Build a lightweight ML baseline (if requested)
  5. Explain recommendations in business terms

---

## 1) 6-Day Crash Revision Roadmap

### Day 1 — Environment + data setup

- Validate environment:
  - `python --version`
  - `pip show pandas numpy scikit-learn matplotlib`
- Load `UOB_sample_dataset.csv` and verify dtypes, nulls, duplicate keys.
- Target output: clean ingestion script + data audit table.

### Day 2 — Data cleaning mastery

- Practice robust cleaning functions:
  - `to_datetime`, `to_numeric(errors='coerce')`, `.fillna()`, `.drop_duplicates()`.
- Build reusable `clean_columns(df)` and `validate_schema(df)` functions.
- Target output: one clean DataFrame and validation checks.

### Day 3 — KPI and groupby analytics

- Build KPI tables by `team`, `hour`, `weekday`, `use_case`.
- Required KPIs:
  - abandon rate
  - SLA breach rate (`queue_time_sec > 120`)
  - avg handle time
  - ATM utilization (`atm_withdrawn / atm_capacity`)
- Target output: ranked KPI dashboard table with top risk segments.

### Day 4 — Merge + feature engineering + anomaly logic

- Join cross-domain data (`ContactCenter`, `ATM`, `Workforce`, `Document`).
- Engineer flags:
  - `sla_breach`, `needs_callback`, `low_ocr`, `high_doc_error`
- Add anomaly detection using z-score/IQR.
- Target output: risk-flagged dataset + automated trigger list.

### Day 5 — Basic ML implementation

- Build one binary target (`anomaly_flag` or `sla_breach`).
- Train Logistic Regression and Random Forest.
- Evaluate with accuracy, precision, recall, confusion matrix.
- Target output: comparison table + feature importance/coefficient summary.

### Day 6 — Full mock simulation

- Attempt one full mock in 90 minutes + 30 minutes explanation rehearsal.
- Use structured explanation: **Problem → Approach → Result → Impact**.
- Target output: interview-ready narrative and reusable code skeleton.

---

## 2) Key Pandas Patterns to Master (with code)

### A. Ingestion + type safety

```python
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    numeric_cols = [
        "queue_time_sec", "handle_time_sec", "atm_capacity",
        "atm_withdrawn", "atm_topup_amt", "ocr_confidence",
        "doc_errors", "workforce_hours", "anomaly_flag"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
```

### B. Cleaning and null handling

```python
def clean_operational_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["agent_id"] = out["agent_id"].fillna("UNASSIGNED")
    out["team"] = out["team"].fillna("UNKNOWN")
    out = out.drop_duplicates(subset=["record_id"], keep="last")
    return out
```

### C. Groupby KPI calculation

```python
def contact_center_kpis(df: pd.DataFrame) -> pd.DataFrame:
    cc = df[df["use_case"] == "ContactCenter"].copy()
    cc["hour"] = cc["timestamp"].dt.hour
    cc["sla_breach"] = cc["queue_time_sec"] > 120

    kpi = (cc.groupby(["team", "hour"], as_index=False)
             .agg(
                 total_calls=("record_id", "count"),
                 avg_queue_sec=("queue_time_sec", "mean"),
                 avg_handle_sec=("handle_time_sec", "mean"),
                 abandon_rate=("abandoned", "mean"),
                 sla_breach_rate=("sla_breach", "mean")
             ))
    return kpi.sort_values(["sla_breach_rate", "abandon_rate"], ascending=False)
```

### D. Merge + feature engineering

```python
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["is_peak_hour"] = out["hour"].isin([8, 9, 10, 17, 18])
    out["sla_breach"] = out["queue_time_sec"] > 120
    out["atm_utilization"] = out["atm_withdrawn"] / out["atm_capacity"]
    out["low_ocr"] = out["ocr_confidence"] < 0.80
    out["high_doc_error"] = out["doc_errors"] >= 2
    return out
```

### E. Anomaly detection (IQR)

```python
def iqr_flags(df: pd.DataFrame, col: str) -> pd.DataFrame:
    x = df[col].dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = df.copy()
    out[f"{col}_anomaly"] = (out[col] < low) | (out[col] > high)
    return out
```

### F. Simple ML baseline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def run_ml_baseline(df):
    ml = df.dropna(subset=["queue_time_sec", "handle_time_sec", "hour", "anomaly_flag"]).copy()
    X = ml[["queue_time_sec", "handle_time_sec", "hour"]]
    y = ml["anomaly_flag"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    logit = LogisticRegression(max_iter=300).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

    print("Logistic Regression")
    print(classification_report(y_test, logit.predict(X_test)))
    print("Random Forest")
    print(classification_report(y_test, rf.predict(X_test)))
```

---

## 3) Common Data Cleaning Scenarios in Banking

- Mixed timestamp formats from multiple systems (`YYYY-MM-DD`, `DD/MM/YYYY`, timezone offsets).
- Missing identifiers (`agent_id`, `atm_id`, `team`) causing dropped rows in joins.
- Numeric fields stored as text (`"1,200"`, `"98%"`, blanks) requiring coercion.
- Duplicate operational records with late updates (same `record_id`, different timestamp/status).
- Inconsistent category values (`Tier-1`, `tier1`, `TIER1`) needing normalization.
- Sparse document-processing columns for non-document rows (must retain row integrity while filtering).

---

## 4) Three Realistic Mock Problems

1. **Contact Center Performance Triage**
   - Compute hourly and team-level SLA breach + abandonment rates.
   - Identify top 3 high-risk windows.
   - Propose an automation rule to reduce queue pressure.

2. **ATM Cash Optimization + Service Risk**
   - Flag ATMs with utilization > 85% and correlate with complaint/queue spikes.
   - Recommend top-up prioritization and expected service impact.

3. **Operational Risk Early Warning**
   - Build a classifier predicting `anomaly_flag` using queue, handle time, and workforce/document indicators.
   - Explain precision-recall tradeoff and operational deployment threshold.

---

## 5) Full Sample Dataset (CSV)

Use this exact CSV for all practice scenarios.

```csv
record_id,use_case,timestamp,queue_time_sec,handle_time_sec,agent_id,resolved,abandoned,channel,team,atm_id,atm_capacity,atm_withdrawn,atm_topup_amt,doc_type,ocr_confidence,doc_errors,workforce_team,workforce_hours,anomaly_flag
C1001,ContactCenter,2026-01-05 08:05:00,15,320,A13,True,False,Voice,Tier1,,,,,,,,,,0
C1002,ContactCenter,2026-01-05 08:12:00,180,0,A07,False,True,Voice,Tier2,,,,,,,,,,1
C1003,ContactCenter,2026-01-05 09:01:00,75,210,A02,True,False,Chat,Tier2,,,,,,,,,,0
C1004,ContactCenter,2026-01-05 09:25:00,32,180,,True,False,Email,Tier1,,,,,,,,,,0
C1005,ContactCenter,2026-01-05 10:10:00,132,400,A05,True,False,Voice,Tier3,,,,,,,,,,1
W001,Workforce,2026-01-05 14:00:00,,,,,,,,,,,,,,,Ops,48,0
W002,Workforce,2026-01-06 02:00:00,,,,,,,,,,,,,,,Ops,40,0
ATM01,ATM,2026-01-05 09:00:00,,,,,,,,ATM01,200000,152000,50000,,,,,,1
ATM02,ATM,2026-01-05 09:30:00,,,,,,,,ATM02,150000,72000,20000,,,,,,0
ATM03,ATM,2026-01-06 04:15:00,,,,,,,,ATM03,180000,178000,45000,,,,,,1
D1001,Document,2026-01-06 11:00:00,,,,,,,,,,,,loan_form,0.94,0,,,0
D1002,Document,2026-01-06 11:20:00,,,,,,,,,,,,claims,0.67,3,,,1
```

---

## 6) Model Answer Outline (Mock 1)

1. **Ingest + validate:** parse datetime, coerce numeric fields, check nulls/duplicates.
2. **Engineer features:** `hour`, `weekday`, `sla_breach`, `needs_callback`.
3. **Build KPI table:** by team + hour with rates and ranking.
4. **Identify hotspot:** highest breach-abandon cluster.
5. **Recommend automation:** callback/workforce trigger rule for that hotspot.
6. **Quantify impact:** expected reduction in breach rate and improved customer wait time.

---

## 7) Business Framing Phrases

- “This segment is the highest operational pain point and should be prioritized first.”
- “The proposed automation reduces manual load while protecting SLA adherence.”
- “These anomalies are not just statistical outliers; they indicate controllable process breaks.”
- “Model output is used as a triage signal, with human review for high-impact cases.”
- “This intervention improves customer experience and lowers avoidable rework.”

---

## 8) Physical Test Readiness Checklist

- Laptop charger, stable Python environment, and local package availability.
- Pre-verified imports: `pandas`, `numpy`, `sklearn`, `matplotlib`.
- One reusable script template with functions for load/clean/kpi/model.
- Comfort with `groupby`, `merge`, datetime ops, and boolean filters under time pressure.
- A concise explanation flow: **what happened, why it matters, what to do next**.

---

## Additional Technical Materials (recommended)

### A. Reusable coding template (create before test)

```python
def main(path="UOB_sample_dataset.csv"):
    df = load_data(path)
    df = clean_operational_data(df)
    feats = build_features(df)
    kpi = contact_center_kpis(feats)
    print(kpi.head())
    run_ml_baseline(feats)

if __name__ == "__main__":
    main()
```

### B. Fast sanity checks

```python
assert df["timestamp"].isna().sum() == 0, "Timestamp parse failed"
assert df["record_id"].nunique() == len(df), "Duplicate record_id detected"
```

### C. Time management for 2-hour test

- 0–20 min: ingest + cleaning
- 20–60 min: KPIs + anomaly logic
- 60–90 min: ML baseline (if asked)
- 90–120 min: interpretation + final summary

### D. Deliverables checklist during interview

- Cleaned dataset + feature table
- KPI summary table sorted by risk
- One automation rule with threshold
- One model output table (if required)
- Business recommendation in 3–5 bullets

### E. Full 2-hour simulation pack

- Assessment brief: `uob_2hr_simulated_assessment.md`
- Simulation dataset: `uob_2hr_assessment_dataset.csv`
- Use this pack when you want a realistic timed run with expected outputs, hidden edge cases, interviewer follow-ups, and a high-scoring solution rubric.
