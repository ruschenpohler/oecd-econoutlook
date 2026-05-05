"""
Phase 1.2 — feature engineering: quarterly lag windows and CLI lag-stacking.

Inputs:
    data/raw_quarterly.parquet  — quarterly panel (GDP + components + KEI)
    data/cli_monthly.parquet    — monthly CLI (amplitude-adjusted index)

Output:
    data/features_quarterly.parquet — wide feature matrix

Spec A (full 37-country): 25 macro lag columns, no CLI.
Spec B (12-country CLI subset): adds 12 CLI lag columns.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_PATH = DATA_DIR / "raw_quarterly.parquet"
CLI_PATH = DATA_DIR / "cli_monthly.parquet"
OUTPUT_PATH = DATA_DIR / "features_quarterly.parquet"

# Target variable columns
TARGET_VARS = ["gdpv_qq_annualised", "gdpv_yy_computed"]

# Macro indicators to lag (excludes GDP — lagged separately)
MACRO_VARS = ["unr", "cbgdpr", "itv_annpct", "xgsv_annpct", "mgsv_annpct"]

LAG_DEPTH = 4  # quarterly lags: 1 through 4

# CLI countries (where headline LI is available)
CLI_COUNTRIES = {
    "AUS",
    "CAN",
    "DEU",
    "ESP",
    "FRA",
    "GBR",
    "ITA",
    "JPN",
    "KOR",
    "MEX",
    "TUR",
    "USA",
}


def create_gdp_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Create gdp_lag1..gdp_lag4 and gdp_accel for the annualised Q/Q GDP growth."""
    out = df.sort_values(["country_code", "year_quarter"]).copy()
    g = out.groupby("country_code")["gdpv_qq_annualised"]
    for lag in range(1, LAG_DEPTH + 1):
        out[f"gdp_lag{lag}"] = g.shift(lag)
    out["gdp_accel"] = out["gdp_lag1"] - out["gdp_lag2"]
    return out


def create_macro_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Create {var}_lag{1..4} for each macroeconomic indicator."""
    out = df.sort_values(["country_code", "year_quarter"]).copy()
    for var in MACRO_VARS:
        if var not in out.columns:
            continue
        g = out.groupby("country_code")[var]
        for lag in range(1, LAG_DEPTH + 1):
            out[f"{var}_lag{lag}"] = g.shift(lag)
    return out


def stack_cli_to_quarterly(cli: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape monthly CLI into quarterly lag columns: cli_q1_m0 .. cli_q4_m2.

    For a target quarter t ending in month M_t (3, 6, 9, 12):
        cli_q{q}_m{m} is CLI for month M_t - 3*q - 2 + m
    where q ∈ {1..4}, m ∈ {0, 1, 2}. No q0 (current quarter).
    """
    # Compute end-month integer for each quarter: e.g. 2024Q1 -> 202403
    dates = panel[["country_code", "year_quarter"]].drop_duplicates().copy()
    dates[["year", "quarter"]] = dates["year_quarter"].str.extract(r"(\d{4})-Q(\d)")
    dates["end_month_int"] = dates["year"].astype(int) * 100 + dates["quarter"].astype(int) * 3

    # Build lookup: (country_code, year_month_int) -> cli
    cli_lookup = cli.copy()
    cli_lookup[["year", "month"]] = cli_lookup["year_month"].str.split("-", expand=True)
    cli_lookup["ym_int"] = cli_lookup["year"].astype(int) * 100 + cli_lookup["month"].astype(int)
    cli_lookup = cli_lookup.set_index(["country_code", "ym_int"])["cli"]

    rows = []
    for q in range(1, LAG_DEPTH + 1):
        for m in range(3):
            offset_months = 3 * q + 2 - m
            temp = dates.copy()
            temp["target_ym"] = temp["end_month_int"].apply(
                lambda em: _sub_months(em, offset_months)
            )
            temp[f"cli_q{q}_m{m}"] = temp.set_index(["country_code", "target_ym"]).index.map(
                cli_lookup
            )
            temp = temp[["country_code", "year_quarter", f"cli_q{q}_m{m}"]]
            rows.append(temp.set_index(["country_code", "year_quarter"]))

    return pd.concat(rows, axis=1).reset_index()


def _sub_months(ym: int, n: int) -> int:
    """Subtract n months from an integer YYYYMM."""
    year, month = divmod(ym, 100)
    total = year * 12 + (month - 1) - n
    new_year, new_month = divmod(total, 12)
    return new_year * 100 + (new_month + 1)


def drop_incomplete_lags(df: pd.DataFrame, lag_depth: int = LAG_DEPTH) -> pd.DataFrame:
    """Drop rows where any lagged feature is NaN (Phase 1 truncated balanced panel)."""
    lag_cols = [c for c in df.columns if "_lag" in c or c == "gdp_accel"]
    return df.dropna(subset=lag_cols)


def main():
    print("=" * 60)
    print("Phase 1.2 — Feature engineering")
    print("=" * 60)

    # Load quarterly panel
    print(f"\n[1/3] Loading quarterly panel from {INPUT_PATH.name} ...")
    panel = pd.read_parquet(INPUT_PATH)
    print(f"  {len(panel)} rows, {panel['country_code'].nunique()} countries")

    # Create macro lags
    print("\n[2/3] Creating lag features ...")
    panel = create_gdp_lags(panel)
    panel = create_macro_lags(panel)

    # Load CLI and stack
    print(f"\n[3/3] Loading CLI from {CLI_PATH.name} and stacking ...")
    cli = pd.read_parquet(CLI_PATH)
    print(f"  {len(cli)} monthly rows, {cli['country_code'].nunique()} countries")
    cli_features = stack_cli_to_quarterly(cli, panel)
    print(f"  CLI feature columns: {[c for c in cli_features.columns if c.startswith('cli_')]}")

    # Merge CLI features into panel (left join — NaN for non-CLI countries)
    panel = panel.merge(cli_features, on=["country_code", "year_quarter"], how="left")

    # Drop incomplete rows (Phase 1 ragged-edge: truncated balanced panel)
    panel = drop_incomplete_lags(panel)
    print(f"\nAfter dropping incomplete lags: {len(panel)} rows")

    # Identify feature columns
    lag_cols = [c for c in panel.columns if "_lag" in c or c == "gdp_accel"]
    cli_cols = [c for c in panel.columns if c.startswith("cli_")]
    print(f"\n  Feature columns: {len(lag_cols)} macro lags + {len(cli_cols)} CLI lags")
    print(f"  Spec A feature count: {len(lag_cols)} (no CLI)")
    print(f"  Spec B feature count: {len(lag_cols) + len(cli_cols)} (with CLI)")

    # Output
    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  Shape: {panel.shape}")
    print(f"  Columns: {len(panel.columns)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
