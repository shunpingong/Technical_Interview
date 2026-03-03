# UOB Mock Problem Set (Banking Intelligent Automation)

## Mock 1 — Contact Center SLA Recovery

### Prompt

Build a KPI table by team and hour. Highlight the highest-risk segment using abandonment rate and SLA breach rate (`queue_time_sec > 120`). Propose one automation intervention.

### Expected deliverables

- Cleaned dataset with engineered flags
- Ranked KPI output table
- One threshold-based automation rule
- 3-bullet business recommendation

### Suggested automation rule

If `sla_breach_rate > 0.25` and `abandon_rate > 0.15` for a team-hour segment, trigger automatic callback queue and temporary staffing alert.

---

## Mock 2 — ATM Cash Optimization

### Prompt

Identify ATM machines with high utilization and map risk to service pressure windows. Recommend which ATMs to top up first.

### Expected deliverables

- ATM utilization table (`atm_withdrawn / atm_capacity`)
- Top-up priority list with rationale
- Optional scenario analysis (e.g., +20% withdrawal stress)

### Suggested threshold

Mark ATM as high risk if utilization > 0.85.

---

## Mock 3 — Operational Risk Early Warning Model

### Prompt

Build a binary classifier to predict `anomaly_flag` using operational features. Explain model trade-offs and deployment recommendation.

### Expected deliverables

- Feature prep pipeline
- Logistic regression + random forest baseline
- Precision/recall summary
- Recommendation for human-in-the-loop threshold

### Suggested framing

Use model output for triage prioritization, not fully automated decisions.
