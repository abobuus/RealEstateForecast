"""
Train Prophet models with macro regressors (cbr_rate, inflation) for all regions.

Inputs:
  data/processed/price_timeseries_2018_2026.csv
  data/macro/Инфляция и ключевая ставка Банка России_F01_01_2018_T28_05_2026.xlsx

Outputs:
  data/processed/macro_quarterly.csv   — quarterly macro data 2018-2026
  models/prophet/region_{code}.pkl     — trained Prophet model per region
  models/prophet/meta.json             — metadata (regions, regressors, date range)
"""

import os, json, pickle
import pandas as pd
import numpy as np
from prophet import Prophet

PROCESSED_DIR = "data/processed"
MACRO_FILE    = "data/macro/Инфляция и ключевая ставка Банка России_F01_01_2018_T28_05_2026.xlsx"
MODELS_DIR    = "models/prophet"
TRAIN_END     = "2023-10-01"


# ── 1. Parse and aggregate macro data to quarterly ───────────────────────────
def load_macro_quarterly() -> pd.DataFrame:
    df = pd.read_excel(MACRO_FILE, sheet_name=0, header=0, engine="openpyxl", dtype={0: str})
    df.columns = ["date_str", "cbr_rate", "inflation", "inflation_target"]

    # Parse date "MM.YYYY" -> datetime
    df["period"] = pd.to_datetime(
        df["date_str"].apply(lambda x: f"{x.split('.')[1]}-{x.split('.')[0].zfill(2)}-01")
    )
    df = df.sort_values("period").reset_index(drop=True)

    # Aggregate to quarters (mean of 3 months)
    df["quarter_start"] = df["period"].dt.to_period("Q").dt.start_time
    macro_q = (
        df.groupby("quarter_start")
        .agg(cbr_rate=("cbr_rate", "mean"), inflation=("inflation", "mean"))
        .reset_index()
        .rename(columns={"quarter_start": "period"})
    )
    macro_q["period"] = pd.to_datetime(macro_q["period"])
    return macro_q


# ── 2. Train Prophet with regressors for one region ─────────────────────────
def train_prophet_region(ts_region: pd.DataFrame, macro_q: pd.DataFrame) -> Prophet | None:
    df = ts_region[["period", "price_sqm"]].copy()
    df = df.rename(columns={"period": "ds", "price_sqm": "y"})
    df = df.merge(macro_q, left_on="ds", right_on="period", how="left").drop(columns=["period"])
    df = df.dropna(subset=["y", "cbr_rate", "inflation"])

    train = df[df["ds"] <= TRAIN_END]
    if len(train) < 12:
        return None

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.3,
    )
    m.add_regressor("cbr_rate")
    m.add_regressor("inflation")
    m.fit(train)
    return m


# ── 3. Main ──────────────────────────────────────────────────────────────────
def main():
    # Macro
    print("Loading macro data...")
    macro_q = load_macro_quarterly()
    macro_q.to_csv(f"{PROCESSED_DIR}/macro_quarterly.csv", index=False)
    print(f"  Quarters: {len(macro_q)}  |  {macro_q['period'].min().date()} - {macro_q['period'].max().date()}")
    print(f"  CBR rate range: {macro_q['cbr_rate'].min():.1f}% - {macro_q['cbr_rate'].max():.1f}%")
    print(f"  Inflation range: {macro_q['inflation'].min():.1f}% - {macro_q['inflation'].max():.1f}%")

    # Time series
    ts = pd.read_csv(f"{PROCESSED_DIR}/price_timeseries_2018_2026.csv", parse_dates=["period"])
    regions = sorted(ts["region"].dropna().astype(int).unique())
    print(f"\nTraining Prophet for {len(regions)} regions...")

    # Region names
    try:
        ref = pd.read_csv("data/russia_region_codes.csv", encoding="utf-8")
        ref["region_code"] = ref["kladr_id"].astype(str).str.zfill(13).str[:2].astype(int)
        reg_names = dict(zip(ref["region_code"], ref["name"]))
    except Exception:
        reg_names = {}

    trained = {}
    failed  = []
    for reg in regions:
        sub = ts[ts["region"] == reg].sort_values("period")
        model = train_prophet_region(sub, macro_q)
        if model is None:
            failed.append(reg)
            continue
        path = f"{MODELS_DIR}/region_{reg}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        trained[reg] = path
        print(f"  [{reg:2d}] {reg_names.get(reg, '?'):30s}  -> saved")

    print(f"\nTrained: {len(trained)}  |  Failed: {len(failed)}")
    if failed:
        print(f"Failed regions: {failed}")

    # Save metadata
    meta = {
        "regions": {str(r): reg_names.get(r, f"Регион {r}") for r in trained},
        "regressors": ["cbr_rate", "inflation"],
        "train_end": TRAIN_END,
        "macro_range": {
            "min": str(macro_q["period"].min().date()),
            "max": str(macro_q["period"].max().date()),
        },
        "cbr_defaults": macro_q[macro_q["period"] >= "2024-01-01"][["period", "cbr_rate", "inflation"]]
            .assign(period=lambda d: d["period"].dt.strftime("%Y-%m-%d"))
            .to_dict("records"),
    }
    with open(f"{MODELS_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved meta -> {MODELS_DIR}/meta.json")


if __name__ == "__main__":
    main()
