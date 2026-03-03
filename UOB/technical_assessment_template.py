import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    numeric_cols = [
        "queue_time_sec", "handle_time_sec", "atm_capacity", "atm_withdrawn",
        "atm_topup_amt", "ocr_confidence", "doc_errors", "workforce_hours", "anomaly_flag",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["agent_id"] = out["agent_id"].fillna("UNASSIGNED")
    out["team"] = out["team"].fillna("UNKNOWN")
    out = out.drop_duplicates(subset=["record_id"], keep="last")
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["weekday"] = out["timestamp"].dt.day_name()
    out["sla_breach"] = out["queue_time_sec"] > 120
    out["needs_callback"] = (out["abandoned"] == True) & (out["resolved"] == False)
    out["atm_utilization"] = out["atm_withdrawn"] / out["atm_capacity"]
    out["low_ocr"] = out["ocr_confidence"] < 0.80
    out["high_doc_error"] = out["doc_errors"] >= 2
    return out


def kpi_table(df: pd.DataFrame) -> pd.DataFrame:
    cc = df[df["use_case"] == "ContactCenter"].copy()
    kpi = (
        cc.groupby(["team", "hour"], as_index=False)
        .agg(
            total_calls=("record_id", "count"),
            avg_queue_sec=("queue_time_sec", "mean"),
            avg_handle_sec=("handle_time_sec", "mean"),
            abandon_rate=("abandoned", "mean"),
            sla_breach_rate=("sla_breach", "mean"),
        )
        .sort_values(["sla_breach_rate", "abandon_rate"], ascending=False)
    )
    return kpi


def run_baseline_ml(df: pd.DataFrame) -> None:
    ml = df.dropna(subset=["queue_time_sec", "handle_time_sec", "hour", "anomaly_flag"]).copy()
    if ml.empty:
        print("No rows available for ML baseline.")
        return

    X = ml[["queue_time_sec", "handle_time_sec", "hour"]]
    y = ml["anomaly_flag"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=300),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"\n{name}")
        print(classification_report(y_test, preds, zero_division=0))


def main(path: str = "UOB_sample_dataset.csv") -> None:
    df = load_data(path)
    df = clean_data(df)
    df = build_features(df)

    print("=== KPI Table (Top Risk Segments) ===")
    print(kpi_table(df).head(10))

    print("\n=== ML Baseline ===")
    run_baseline_ml(df)


if __name__ == "__main__":
    main()
