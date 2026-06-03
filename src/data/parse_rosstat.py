"""
Parse Rosstat housing price files (2021-2025) and combine with our 2018-2021 data.

Outputs:
  data/rosstat/rosstat_prices_quarterly.csv  — parsed Rosstat data
  data/processed/price_timeseries_2018_2025.csv — full joined time series
"""

import re
import pandas as pd
import numpy as np

ROSSTAT_DIR = "data/rosstat"
PROCESSED_DIR = "data/processed"

# Files to parse: (year, market_type, filename)
ROSSTAT_FILES = [
    (2021, "perv", "sred_cen_perv_4kv-2021.xlsx"),
    (2022, "perv", "sred_cen_perv_4kv-2022_.xlsx"),
    (2023, "perv", "sred_cen_perv_4kv-2023.xlsx"),
    (2024, "perv", "sred_cen_perv_4kv-2024.xlsx"),
    (2025, "perv", "sred_cen_perv_4kv-2025.xlsx"),
    (2021, "vtor", "sred_cen_vtor_4kv-2021.xlsx"),
    (2022, "vtor", "sred_cen_vtor_4kv-2022_.xlsx"),
    (2023, "vtor", "sred_cen_vtor_4kv-2023.xlsx"),
    (2024, "vtor", "sred_cen_vtor_4kv-2024.xlsx"),
    (2025, "vtor", "sred_cen_vtor_4kv-2025.xlsx"),
]

# Quarter columns: perv=[1,5,9,13], vtor=[1,6,11,16]
Q_COLS = {"perv": [1, 5, 9, 13], "vtor": [1, 6, 11, 16]}


def extract_region_code(cell_value: str) -> int | None:
    """
    Extract 2-digit region code from Rosstat cell like '77000000000 - г. Москва'.
    Returns None for Russia total (643) and federal districts (030, 040, ...).
    """
    if not isinstance(cell_value, str):
        return None
    cell_value = cell_value.strip()
    # Get numeric part before ' - '
    m = re.match(r"^(\d+)", cell_value)
    if not m:
        return None
    code_str = m.group(1)
    # Keep only 11-digit regional codes (ends in 9 zeros)
    if len(code_str) == 11 and code_str[2:] == "000000000":
        return int(code_str[:2])
    return None


def parse_file(year: int, market: str, filename: str) -> pd.DataFrame:
    """Parse one Rosstat Excel file into tidy DataFrame."""
    path = f"{ROSSTAT_DIR}/{filename}"
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[1]  # first data sheet (index 1, skip contents)
    df_raw = pd.read_excel(path, sheet_name=sheet, header=None)

    # Find data start row
    data_start = None
    for i, row in df_raw.iterrows():
        if re.match(r"^\d{2,}", str(row.iloc[0]).strip()):
            data_start = i
            break
    if data_start is None:
        print(f"  WARNING: could not find data start in {filename}")
        return pd.DataFrame()

    q_cols = Q_COLS[market]
    rows = []
    for i in range(data_start, len(df_raw)):
        row = df_raw.iloc[i]
        region_code = extract_region_code(str(row.iloc[0]))
        if region_code is None:
            continue
        for q_idx, col in enumerate(q_cols, start=1):
            val = row.iloc[col]
            # Convert to float, skip missing ('-', '...', NaN)
            try:
                price = float(val)
                if np.isnan(price):
                    price = None
            except (ValueError, TypeError):
                price = None
            rows.append({
                "year": year,
                "quarter": q_idx,
                "region": region_code,
                "market": market,
                "price_sqm_rosstat": price,
            })

    return pd.DataFrame(rows)


def main():
    # ── 1. Parse all Rosstat files ───────────────────────────────────────────
    print("=== Parsing Rosstat files ===")
    frames = []
    for year, market, fname in ROSSTAT_FILES:
        df = parse_file(year, market, fname)
        n = len(df[df["price_sqm_rosstat"].notna()])
        print(f"  {fname}: {len(df)} rows, {n} non-null prices, "
              f"{df['region'].nunique()} unique regions")
        frames.append(df)

    rosstat = pd.concat(frames, ignore_index=True)

    # ── 2. Validate region codes ─────────────────────────────────────────────
    print("\n=== Region code validation ===")
    rosstat_regions = sorted(rosstat["region"].unique())
    print(f"Rosstat regions ({len(rosstat_regions)}): {rosstat_regions}")

    # Load our region list for cross-check
    try:
        our_regions = pd.read_csv(f"{PROCESSED_DIR}/regions_stats.csv")["region"].astype(int).tolist()
        our_set = set(our_regions)
        rosstat_set = set(rosstat_regions)
        only_rosstat = sorted(rosstat_set - our_set)
        only_ours = sorted(our_set - rosstat_set)
        in_both = sorted(our_set & rosstat_set)
        print(f"In both datasets: {len(in_both)} regions: {in_both}")
        print(f"Only in Rosstat (no transactions): {only_rosstat}")
        print(f"Only in our data (not in Rosstat): {only_ours}")
    except FileNotFoundError:
        print("  regions_stats.csv not found, skipping cross-check")

    # ── 3. Save Rosstat data ─────────────────────────────────────────────────
    out_rosstat = f"{ROSSTAT_DIR}/rosstat_prices_quarterly.csv"
    rosstat.to_csv(out_rosstat, index=False)
    print(f"\nSaved {len(rosstat):,} rows -> {out_rosstat}")

    # ── 4. Combine with our 2018-2021 data ───────────────────────────────────
    print("\n=== Building combined time series ===")

    # Load our monthly data and aggregate to quarters
    our_monthly = pd.read_csv(f"{PROCESSED_DIR}/price_by_region_month.csv", parse_dates=["period"])
    our_monthly["quarter"] = our_monthly["period"].dt.quarter
    our_monthly["year"] = our_monthly["period"].dt.year

    our_q = (
        our_monthly.groupby(["year", "quarter", "region"])
        .agg(
            price_sqm_ours=("median_price_sqm", "median"),
            n_deals=("count", "sum"),
        )
        .reset_index()
    )
    print(f"Our data (2018-2021, quarterly): {len(our_q):,} rows, "
          f"{our_q['region'].nunique()} regions")

    # Average perv+vtor from Rosstat into one price
    rosstat_avg = (
        rosstat.groupby(["year", "quarter", "region"])["price_sqm_rosstat"]
        .mean()
        .reset_index()
    )

    # Merge: use our data for 2018-2021, Rosstat for 2022-2025
    # For 2021 overlap: keep both for comparison
    our_q["source"] = "transactions"
    rosstat_avg["source"] = "rosstat"
    rosstat_avg = rosstat_avg.rename(columns={"price_sqm_rosstat": "price_sqm"})
    our_q = our_q.rename(columns={"price_sqm_ours": "price_sqm"})

    # 2018-2020: only our data
    part1 = our_q[our_q["year"] <= 2020][["year", "quarter", "region", "price_sqm", "source"]]

    # 2021: average of our + Rosstat (overlap year for sanity check)
    ours_2021 = our_q[our_q["year"] == 2021][["year", "quarter", "region", "price_sqm"]].copy()
    ours_2021["source"] = "transactions"
    ross_2021 = rosstat_avg[rosstat_avg["year"] == 2021][["year", "quarter", "region", "price_sqm"]].copy()
    ross_2021["source"] = "rosstat"

    # Print 2021 overlap comparison for key regions
    print("\n--- 2021 overlap check (transactions vs Rosstat) ---")
    key_regions = [77, 78, 54, 23, 50]
    merged_2021 = ours_2021.merge(ross_2021, on=["year", "quarter", "region"],
                                   suffixes=("_ours", "_rosstat"))
    merged_2021["diff_pct"] = ((merged_2021["price_sqm_rosstat"] - merged_2021["price_sqm_ours"])
                                / merged_2021["price_sqm_ours"] * 100).round(1)
    check = merged_2021[merged_2021["region"].isin(key_regions)].sort_values(["region", "quarter"])
    for _, r in check.iterrows():
        print(f"  Region {r['region']:2.0f} Q{r['quarter']:.0f}: "
              f"ours={r['price_sqm_ours']:,.0f}  rosstat={r['price_sqm_rosstat']:,.0f}  "
              f"diff={r['diff_pct']:+.1f}%")

    # Use Rosstat for 2021 in final series (official data)
    part2 = ross_2021.copy()
    part2["source"] = "rosstat"

    # 2022-2025: Rosstat only
    part3 = rosstat_avg[rosstat_avg["year"] >= 2022][["year", "quarter", "region", "price_sqm", "source"]]

    combined = pd.concat([part1, part2, part3], ignore_index=True)
    combined = combined.sort_values(["region", "year", "quarter"]).reset_index(drop=True)

    # Add period column (first month of each quarter)
    combined["period"] = pd.to_datetime(
        combined["year"].astype(str) + "-" +
        ((combined["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01"
    )

    # Filter to regions present in both our data and Rosstat
    common_regions = set(our_q["region"].unique()) & set(rosstat_avg["region"].unique())
    combined = combined[combined["region"].isin(common_regions)]

    print(f"\nCombined time series: {len(combined):,} rows")
    print(f"Regions: {combined['region'].nunique()}")
    print(f"Period: {combined['period'].min().date()} - {combined['period'].max().date()}")
    print(f"Quarters per region: {combined.groupby('region').size().describe().to_dict()}")

    out_ts = f"{PROCESSED_DIR}/price_timeseries_2018_2025.csv"
    combined.to_csv(out_ts, index=False)
    print(f"\nSaved {len(combined):,} rows -> {out_ts}")


if __name__ == "__main__":
    main()
