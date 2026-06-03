"""
Parse Rosstat quarterly price index files (ind_perv / ind_vtor, 2021-2026).

Outputs:
  data/rosstat/rosstat_indices_quarterly.csv  — raw quarterly indices per region/market
  data/processed/price_timeseries_2018_2026.csv — full time series (our data + rosstat chain)
"""

import re
import pandas as pd
import numpy as np

ROSSTAT_DIR  = "data/rosstat"
PROCESSED_DIR = "data/processed"

# (year, quarters_available, market, filename)
# quarters_available: number of full quarters in this file (1 or 4)
INDEX_FILES = [
    (2021, 4, "perv", "ind_perv_2021 г.(1).xlsx"),
    (2021, 4, "vtor", "ind_vtor_2021.xlsx"),
    (2022, 4, "perv", "ind_perv_4kv-2022.xlsx"),
    (2022, 4, "vtor", "ind_vtor_4kv-2022.xlsx"),
    (2023, 4, "perv", "ind_perv_4kv-2023.xlsx"),
    (2023, 4, "vtor", "ind_vtor_4kv-2023.xlsx"),
    (2024, 4, "perv", "ind_perv_4kv-2024(1).xlsx"),
    (2024, 4, "vtor", "ind_vtor_4kv-2024.xlsx"),
    (2025, 4, "perv", "ind_perv_4kv-2025.xlsx"),
    (2025, 4, "vtor", "ind_vtor_4kv-2025.xlsx"),
    (2026, 1, "perv", "ind_perv_1kv-2026.xlsx"),
    (2026, 1, "vtor", "ind_vtor_1kv-2026.xlsx"),
]

# Column index of "все типы" for each quarter (0-indexed, 4 cols per quarter)
# Q1=col1, Q2=col5, Q3=col9, Q4=col13
ALL_TYPES_COLS = {1: 1, 2: 5, 3: 9, 4: 13}


def build_okato_map() -> dict[int, int]:
    """Build OKATO (11-digit int) -> region code (2-digit int) mapping."""
    ref = pd.read_csv("data/russia_region_codes.csv", encoding="utf-8")
    ref["region_code"] = ref["kladr_id"].astype(str).str.zfill(13).str[:2].astype(int)
    return dict(zip(ref["okato"].astype(int), ref["region_code"]))


def extract_okato(cell_value: str) -> int | None:
    """Extract 11-digit OKATO code from Rosstat cell like '45000000000 - г. Москва'."""
    if not isinstance(cell_value, str):
        return None
    m = re.match(r"^(\d+)", cell_value.strip())
    if not m:
        return None
    code_str = m.group(1)
    if len(code_str) == 11 and code_str[2:] == "000000000":
        return int(code_str)
    return None


def parse_index_file(year: int, n_quarters: int, market: str, filename: str,
                     okato_map: dict) -> pd.DataFrame:
    """Parse one index file -> tidy DataFrame with columns:
       year, quarter, region, market, index_pct (% to previous quarter)
    """
    path = f"{ROSSTAT_DIR}/{filename}"
    xl  = pd.ExcelFile(path, engine="openpyxl")
    df  = pd.read_excel(path, sheet_name=xl.sheet_names[1], header=None, engine="openpyxl")

    # Find first data row (starts with 11-digit code)
    data_start = None
    for i, row in df.iterrows():
        if extract_okato(str(row.iloc[0])) is not None:
            data_start = i
            break
    if data_start is None:
        print(f"  WARNING: no data in {filename}")
        return pd.DataFrame()

    quarters = range(1, n_quarters + 1)
    rows = []
    for i in range(data_start, len(df)):
        cell = str(df.iloc[i, 0])
        okato = extract_okato(cell)
        if okato is None:
            continue
        region = okato_map.get(okato)
        if region is None:
            continue
        for q in quarters:
            col = ALL_TYPES_COLS[q]
            val = df.iloc[i, col]
            try:
                idx = float(val)
                if np.isnan(idx) or idx == 0:
                    idx = None
            except (ValueError, TypeError):
                idx = None
            rows.append({
                "year": year, "quarter": q,
                "region": region, "market": market,
                "index_pct": idx,
            })

    return pd.DataFrame(rows)


def main():
    okato_map = build_okato_map()
    print(f"OKATO map: {len(okato_map)} regions")

    # ── 1. Parse all index files ─────────────────────────────────────────────
    print("\n=== Parsing index files ===")
    frames = []
    for year, n_q, market, fname in INDEX_FILES:
        df = parse_index_file(year, n_q, market, fname, okato_map)
        n = len(df[df["index_pct"].notna()])
        print(f"  {fname}: {n} non-null indices, {df['region'].nunique()} regions")
        frames.append(df)

    indices = pd.concat(frames, ignore_index=True)
    indices = indices.sort_values(["region", "year", "quarter", "market"]).reset_index(drop=True)

    out_idx = f"{ROSSTAT_DIR}/rosstat_indices_quarterly.csv"
    indices.to_csv(out_idx, index=False)
    print(f"\nSaved {len(indices):,} rows -> {out_idx}")

    # ── 2. Average perv + vtor per region/quarter ────────────────────────────
    idx_avg = (
        indices.groupby(["year", "quarter", "region"])["index_pct"]
        .mean()
        .reset_index()
        .rename(columns={"index_pct": "index_avg"})
    )

    # ── 3. Validation: compare 2021 indices with our transaction growth ──────
    print("\n=== 2021 validation: index growth vs our transaction data ===")
    our_q = pd.read_csv(f"{PROCESSED_DIR}/price_by_region_month.csv", parse_dates=["period"])
    our_q["quarter"] = our_q["period"].dt.quarter
    our_q["year"]    = our_q["period"].dt.year
    our_qtr = (
        our_q.groupby(["year", "quarter", "region"])["median_price_sqm"]
        .median().reset_index()
    )

    # Quarter-over-quarter growth in our data for 2021
    our_2021 = our_qtr[our_qtr["year"] == 2021].copy()
    our_2020q4 = our_qtr[(our_qtr["year"] == 2020) & (our_qtr["quarter"] == 4)][["region", "median_price_sqm"]].rename(columns={"median_price_sqm": "base_price"})
    our_2021 = our_2021.merge(our_2020q4, on="region", how="inner")
    our_2021["our_growth_pct"] = our_2021["median_price_sqm"] / our_2021["base_price"] * 100

    idx_2021 = idx_avg[idx_avg["year"] == 2021].copy()
    check = our_2021[our_2021["quarter"] == 1].merge(idx_2021[idx_2021["quarter"] == 1], on=["year", "quarter", "region"])

    key_regions = [77, 78, 54, 50, 23, 66, 61, 63]
    check_key = check[check["region"].isin(key_regions)].sort_values("region")
    print(f"{'Region':>8}  {'Ours %':>8}  {'Rosstat %':>10}  {'Diff':>6}")
    for _, r in check_key.iterrows():
        diff = r["index_avg"] - r["our_growth_pct"] if r["index_avg"] else float("nan")
        print(f"{r['region']:>8.0f}  {r['our_growth_pct']:>8.1f}  {r['index_avg']:>10.1f}  {diff:>+6.1f}")

    # ── 4. Build combined time series 2018 Q1 – 2026 Q1 ────────────────────
    print("\n=== Building combined time series ===")

    # Our quarterly prices (2018-2021)
    our_base = our_qtr.copy()
    our_base["source"] = "transactions"
    our_base = our_base.rename(columns={"median_price_sqm": "price_sqm"})

    # Find Q4 2021 base price per region
    base_q4_2021 = our_base[(our_base["year"] == 2021) & (our_base["quarter"] == 4)][
        ["region", "price_sqm"]
    ].copy()

    # Chain indices from Q1 2022 onwards
    chain_idx = idx_avg[idx_avg["year"] >= 2022].sort_values(["region", "year", "quarter"])

    chained_rows = []
    for region, grp in chain_idx.groupby("region"):
        base_row = base_q4_2021[base_q4_2021["region"] == region]
        if len(base_row) == 0:
            continue
        price = float(base_row["price_sqm"].iloc[0])
        for _, row in grp.iterrows():
            if row["index_avg"] is None or np.isnan(row["index_avg"]):
                continue
            price = price * row["index_avg"] / 100.0
            chained_rows.append({
                "year": int(row["year"]),
                "quarter": int(row["quarter"]),
                "region": int(region),
                "price_sqm": price,
                "source": "rosstat_index",
            })

    chained = pd.DataFrame(chained_rows)

    # Use our data for 2018-2021, rosstat chain for 2022+
    part1 = our_base[["year", "quarter", "region", "price_sqm", "source"]].copy()
    combined = pd.concat([part1, chained], ignore_index=True)
    combined = combined.sort_values(["region", "year", "quarter"]).reset_index(drop=True)

    # Add period column (first month of each quarter)
    combined["period"] = pd.to_datetime(
        combined["year"].astype(str) + "-" +
        ((combined["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01"
    )

    # Keep only regions present in both our data and index data
    common = set(our_base["region"].unique()) & set(chained["region"].unique())
    combined = combined[combined["region"].isin(common)]

    print(f"Combined rows:  {len(combined):,}")
    print(f"Regions:        {combined['region'].nunique()}")
    print(f"Period:         {combined['period'].min().date()} - {combined['period'].max().date()}")
    pts_per_region = combined.groupby("region").size()
    print(f"Points/region:  min={pts_per_region.min()}, median={pts_per_region.median():.0f}, max={pts_per_region.max()}")

    out_ts = f"{PROCESSED_DIR}/price_timeseries_2018_2026.csv"
    combined.to_csv(out_ts, index=False)
    print(f"\nSaved {len(combined):,} rows -> {out_ts}")


if __name__ == "__main__":
    main()
