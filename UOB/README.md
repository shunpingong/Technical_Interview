# UOB Intelligent Automation Interview Prep

This folder houses everything you need for the on-site 2-hour Python/Pandas/ML assessment—laptop-ready notebooks, the synthetic dataset, and this structured revision guide that keeps the practice grounded in Contact Center Analytics, Workforce, ATM cash, operational risk, and document processing.

## 1. Revision Roadmap (6-day crash plan)

1. **Day 1 – Environment & data orientation**: Validate Python 3.11+, pip-install `pandas`, `numpy`, `scikit-learn`, `matplotlib`. Open the notebook, load `UOB_sample_dataset.csv`, and skim all columns/rows.
2. **Day 2 – Cleaning & datetime fluency**: Practice coercing types, parsing timestamps, filling `agent_id` or `atm_id`, and flagging missing values/anomalous zeros for queue/handle times.
3. **Day 3 – Groupby + KPI storytelling**: Build aggregated KPIs by team, shift, and ATM (abandon rate, SLA breach, utilization) and sort to highlight the top risk cohort.
4. **Day 4 – Automation + anomaly detection**: Engineer boolean flags, compose reusable filter functions (e.g., SLA breach + zero handle), and detect queue-time outliers with z-scores or rolling medians.
5. **Day 5 – Modular ML & explanation**: Fit logistic regression and random forest models, log key metrics, and practice narrating why feature coefficients/importance matter to ops.
6. **Day 6 – Mock test walkthrough**: Time yourself solving the mock problems below, build the plots or tables you would show, and rehearse the operational impact pitch.

## 2. Key Pandas patterns to master

**a. Groupby + KPI scaffolding**

```python
import pandas as pd

calls = (
    pd.read_csv("UOB_sample_dataset.csv", parse_dates=["timestamp"], keep_default_na=False)
    .query("use_case == 'ContactCenter'")
)

kpi = (
    calls.assign(weekday=calls["timestamp"].dt.day_name())
    .groupby(["team", "weekday"], as_index=False)
    .agg(
        total_calls=("record_id", "count"),
        avg_queue=("queue_time_sec", "mean"),
        sla_breach_pct=("queue_time_sec", lambda q: (q > 120).mean() * 100),
        abandon_rate=("abandoned", "mean"),
    )
    .assign(avg_queue=lambda df: df["avg_queue"].round(1))
)
```

**b. Merge + cross-use-case joins**

```python
calls = pd.read_csv("UOB_sample_dataset.csv", parse_dates=["timestamp"])
atm = calls.query("use_case == 'ATM'")
workforce = calls.query("use_case == 'Workforce'")

kpi = (
    calls[~calls["atm_id"].notna()]
    .merge(atm[["atm_id", "atm_capacity", "atm_withdrawn"]], how="left", on="atm_id")
)
```

**c. Datetime handling & feature engineering**

```python
calls = calls.assign(
    hour=calls["timestamp"].dt.hour,
    is_peak=calls["timestamp"].dt.hour.isin([8, 9, 10, 17, 18])
)
```

**d. Anomaly detection**

```python
from scipy.stats import zscore

def flag_outliers(df, col="queue_time_sec", threshold=3):
    mask = (zscore(df[col].dropna()) > threshold) | (zscore(df[col].dropna()) < -threshold)
    return df.loc[mask.fillna(False)]
```

**e. Modular ML quick-hit**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

features = calls.dropna(subset=["queue_time_sec", "handle_time_sec"]).assign(
    sla_breach=lambda df: (df["queue_time_sec"] > 120).astype(int)
)

X = features[["queue_time_sec", "handle_time_sec", "hour"]]
y = features["sla_breach"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

models = {
    "logistic": LogisticRegression(max_iter=200).fit(X_train, y_train),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train),
}
for name, model in models.items():
    print(name, model.score(X_test, y_test))
```

## 3. Common data cleaning scenarios in banking datasets

- Normalize timestamps from different sources, drop timezone noise, and ensure `timestamp` is in UTC for consistent KPIs.
- Cast numeric columns (`queue_time_sec`, `handle_time_sec`, `atm_capacity`) to `int` after removing stray commas or textual suffixes.
- Fill `agent_id` / `atm_id` gaps with `UNASSIGNED` or placeholder tiers so joins do not drop rows.
- Deduplicate `record_id` before aggregations; prefer `last` or `min` for conflicting timestamps.
- Convert document processing scores (`ocr_confidence`, `doc_errors`) into categorical flags (`high_error`, `low_confidence`) for model explainability.

## 4. Three mock problem statements

1. **Contact Center KPI dashboard** – Aggregate contacts by team and shift, compute SLA breach %, average handle / queue volume, and suggest an automation rule that reduces the busiest breach window.
2. **ATM cash & callback alignment** – Merge ATM withdrawal summaries with contact volume per region, flag top two cash-constraint machines, and simulate the weekly replenishment trigger logic.
3. **Operational Risk classifier** – Engineer features from workforce hours, document flags, and call abandonment to predict whether a case will escalate; include a short explanation of feature importances.

## 5. Sample dataset (`UOB_sample_dataset.csv`)

Use this CSV as the canonical dataset for all practice problems. Each row blends contact center, ATM, workforce, and document processing contexts.

```csv
record_id,use_case,timestamp,queue_time_sec,handle_time_sec,agent_id,resolved,abandoned,channel,team,atm_id,atm_capacity,atm_withdrawn,atm_topup_amt,doc_type,ocr_confidence,doc_errors,workforce_team,workforce_hours,anomaly_flag
C1001,ContactCenter,2026-01-05 08:05:00,15,320,A13,True,False,Voice,Tier1,,,,,,,,,,0
C1002,ContactCenter,2026-01-05 08:12:00,180,0,A07,False,True,Voice,Tier2,,,,,,,,,,1
C1003,ContactCenter,2026-01-05 09:01:00,75,210,A02,True,False,Chat,Tier2,,,,,,,,,,0
C1004,ContactCenter,2026-01-05 09:25:00,32,180,,True,False,Email,Tier1,,,,,,,,,,0
C1005,ContactCenter,2026-01-05 10:10:00,132,400,A05,True,False,Voice,Tier3,,,,,,,,,,1
W001,Workforce,2026-01-05 14:00:00,,,,,,,,,Ops,,, , , , ,48,0
W002,Workforce,2026-01-06 02:00:00,,,,,,,,,Ops,,, , , , ,40,0
ATM01,ATM,2026-01-05 09:00:00,,,,,,,,ATM01,200000,152000,50000,,,,,,1
ATM02,ATM,2026-01-05 09:30:00,,,,,,,,ATM02,150000,72000,20000,,,,,,0
ATM03,ATM,2026-01-06 04:15:00,,,,,,,,ATM03,180000,178000,45000,,,,,,1
D1001,Document,2026-01-06 11:00:00,,,,,,,,,,,,loan_form,0.94,0,,,0
D1002,Document,2026-01-06 11:20:00,,,,,,,,,,,,claims,0.67,3,,,1
```

Run `pd.read_csv("UOB_sample_dataset.csv", parse_dates=["timestamp"])` before each mock problem to ensure dtype consistency.

## 6. Model answer outline (mock 1: Contact Center KPI dashboard)

1. **Ingest** – Load CSV, parse timestamps, fill `agent_id` with `UNASSIGNED`, coerce `queue_time_sec`/`handle_time_sec` to `int`.
2. **Feature engineering** – Add `weekday`, `hour`, boolean `sla_breach` (`queue_time_sec > 120`), and `need_callback` (`abandoned & not resolved`).
3. **Grouping + KPI layer** – Group by `team` and `weekday`, compute total calls, SLA breach %, average handle/queue, and rank by SLA breach percent.
4. **Automation idea** – Pick the worst wallet (e.g., Tier2 afternoons), describe automated callback or workforce spike that would shave breach from `X%` to `Y%`.
5. **Narrative** – Conclude with business impact such as “Focusing automation on the Tier2 afternoon breach reduces wasted handle time and frees Tier1 staff for advisory tasks.”

## 7. Key business framing phrases

- “This insight prioritizes the cohort that’s dragging customer satisfaction scores down.”
- “Automating a callback at the SLA breach spike helps us redeploy limited agents.”
- “These KPIs show where operational risk is concentrated, so the compliance team can focus.”
- “The anomaly score flags where we should investigate whether a workflow or tool is failing.”
- “Using logistic regression/random forest keeps the model explainable for the risk committee.”

## 8. Pre-test checklist

- Confirm Python 3.11+, pandas, numpy, scikit-learn, and matplotlib are installed locally.
- Open `UOB_revision_guide.ipynb` and run each section once to surface dependency or data issues.
- Load the CSV, inspect `calls.info()`, and save a screenshot/note of dtype results for quick reference.
- Practice framing insights with the phrases above so explanations stay clinical and impactful.
- Bring headphones, a timer, and a clean notebook/IDE workspace for the 2-hour block.
