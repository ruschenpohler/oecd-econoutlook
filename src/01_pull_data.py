"""
Phase 1 data pull: QNA (level + growth) + CLI + STES.

Reads from locally downloaded OECD Data Explorer CSV exports.
One entry point, one output file.

Usage:
    uv run python src/01_pull_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

from data_quality import compute_growth_from_level, reconcile_growth, log_reconciliation_summary

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_PATH = DATA_DIR / "raw_quarterly.parquet"
COVERAGE_PATH = OUTPUT_DIR / "data_coverage.md"

# ---------------------------------------------------------------------------
# Downloaded CSV files (manual export from https://data-explorer.oecd.org/)
# ---------------------------------------------------------------------------
QNA_LEVEL_FILE   = DATA_DIR / "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,1.1+all.csv"
QNA_GROWTH_FILE  = DATA_DIR / "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,+all.csv"
CLI_FILE         = DATA_DIR / "OECD.SDD.STES,DSD_STES@DF_CLI,4.1+all.csv"

# STES key short-term indicators — not yet downloaded; skip if missing
STES_FILE        = DATA_DIR / "OECD.SDD.STES,DSD_STES@DF_STES,4.1+all.csv"

# ---------------------------------------------------------------------------
# 38 OECD member country codes (from v1)
# ---------------------------------------------------------------------------
TARGET_COUNTRIES = {
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DEU", "DNK",
    "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HUN", "IRL", "ISL", "ISR",
    "ITA", "JPN", "KOR", "LTU", "LUX", "LVA", "MEX", "NLD", "NOR", "NZL",
    "POL", "PRT", "SVK", "SVN", "SWE", "CHE", "TUR", "USA",
}

# ---------------------------------------------------------------------------
# QNA level series
# ---------------------------------------------------------------------------

def pull_qna_level(path: Path) -> pd.DataFrame:
    """
    Read QNA National Currency CSV, extract real GDP chain-linked level.

    Returns DataFrame with columns:
        country_code, year_quarter, gdp_level_real
    """
    df = pd.read_csv(path, low_memory=False)

    # Filter: GDP (B1GQ), chain-linked (L), seasonally adjusted (Y),
    #         total economy (S1), quarterly (Q), national currency (XDC)
    mask = (
        (df["TRANSACTION"] == "B1GQ")
        & (df["PRICE_BASE"] == "L")
        & (df["ADJUSTMENT"] == "Y")
        & (df["SECTOR"] == "S1")
        & (df["FREQ"] == "Q")
        & (df["UNIT_MEASURE"] == "XDC")
    )
    df = df.loc[mask, ["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["country_code", "year_quarter", "gdp_level_real"]

    # Normalise period format: "1947-Q1" is already correct
    # Filter to target countries
    available = set(df["country_code"].unique())
    missing = TARGET_COUNTRIES - available
    if missing:
        print(f"WARNING: QNA level missing for {sorted(missing)}")
    df = df[df["country_code"].isin(TARGET_COUNTRIES)].copy()

    df["gdp_level_real"] = pd.to_numeric(df["gdp_level_real"], errors="coerce")
    df = df.dropna(subset=["gdp_level_real"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# QNA growth series (for reconciliation)
# ---------------------------------------------------------------------------

def pull_qna_growth(path: Path) -> pd.DataFrame:
    """
    Read QNA GROWTH CSV, extract OECD-published Q/Q (G1) and Y/Y (GY) growth.

    Returns DataFrame with columns:
        country_code, year_quarter, gdpv_qq_published, gdpv_yy_published
    """
    df = pd.read_csv(path, low_memory=False)

    mask = (
        (df["TRANSACTION"] == "B1GQ")
        & (df["ADJUSTMENT"] == "Y")
        & (df["TRANSFORMATION"].isin(["G1", "GY"]))
    )
    df = df.loc[mask, ["REF_AREA", "TIME_PERIOD", "TRANSFORMATION", "OBS_VALUE"]].copy()
    df.columns = ["country_code", "year_quarter", "transformation", "obs_value"]

    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce")

    # Pivot: one column for G1 (Q/Q), one for GY (Y/Y)
    pivoted = df.pivot_table(
        index=["country_code", "year_quarter"],
        columns="transformation",
        values="obs_value",
    ).reset_index()

    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"G1": "gdpv_qq_published", "GY": "gdpv_yy_published"})

    # Filter to target countries
    available = set(pivoted["country_code"].unique())
    missing = TARGET_COUNTRIES - available
    if missing:
        print(f"WARNING: QNA growth missing for {sorted(missing)}")
    pivoted = pivoted[pivoted["country_code"].isin(TARGET_COUNTRIES)].copy()

    return pivoted.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI series
# ---------------------------------------------------------------------------

def pull_cli(path: Path) -> pd.DataFrame:
    """
    Read CLI CSV, extract amplitude-adjusted index (LI + AA + IX).

    Returns long-format DataFrame with columns:
        country_code, year_month, cli
    """
    df = pd.read_csv(path, low_memory=False)

    mask = (
        (df["MEASURE"] == "LI")
        & (df["ADJUSTMENT"] == "AA")
        & (df["TRANSFORMATION"] == "IX")
    )
    df = df.loc[mask, ["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["country_code", "year_month", "cli"]

    # "year_month" is like "2025-06" — normalise to period
    df["year_month"] = df["year_month"].astype(str)
    df["cli"] = pd.to_numeric(df["cli"], errors="coerce")
    df = df.dropna(subset=["cli"])

    # Filter to target countries
    available = set(df["country_code"].unique())
    oecd_with_cli = available & TARGET_COUNTRIES
    oecd_without_cli = TARGET_COUNTRIES - available
    print(f"  CLI available for {len(oecd_with_cli)} OECD members: {sorted(oecd_with_cli)}")
    print(f"  CLI not available for {len(oecd_without_cli)} OECD members (no headline LI series)")
    df = df[df["country_code"].isin(available & TARGET_COUNTRIES)].copy()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STES key short-term indicators — deferred; stub
# ---------------------------------------------------------------------------

def pull_stes(path: Path) -> pd.DataFrame:
    """
    Read STES CSV, extract UNR, CBGDPR, ITV_ANNPCT, XGSV_ANNPCT, MGSV_ANNPCT.

    Returns long-format DataFrame with columns:
        country_code, year_quarter, unr, cbgdpr, ...
    """
    raise NotImplementedError("STES file not yet downloaded")


# ---------------------------------------------------------------------------
# Merge and output
# ---------------------------------------------------------------------------

def log_coverage(df: pd.DataFrame, path: Path) -> None:
    """Write per-country coverage log (start_quarter, end_quarter, n_obs)."""
    coverage = (
        df.groupby("country_code")["year_quarter"]
        .agg(start="min", end="max", n_obs="count")
        .sort_values("start")
    )
    late_entry = coverage[coverage["start"] > "2005-Q4"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# QNA quarterly coverage by country\n\n")
        dropped = TARGET_COUNTRIES - set(coverage.index)
        if dropped:
            f.write(f"**Dropped countries** (missing chain-linked quarterly GDP level): "
                    f"{', '.join(sorted(dropped))}\n\n")
            f.write(f"These countries are available in the GROWTH dataset (published rates) "
                    f"but lack chain-linked (PRICE_BASE=L) levels in the national currency table. "
                    f"Panel proceeds with N={len(coverage)} countries.\n\n")
        f.write(f"Panel: {coverage['start'].min()} to {coverage['end'].max()}, "
                f"N={len(coverage)} countries, {coverage['n_obs'].sum()} observations\n\n")
        f.write("| country_code | start | end | n_obs | late_entry |\n")
        f.write("|--------------|-------|-----|-------|------------|\n")
        for country, row in coverage.iterrows():
            late = "LATE" if row["start"] > "2005-Q4" else ""
            f.write(f"| {country} | {row['start']} | {row['end']} | {row['n_obs']} | {late} |\n")

    if len(late_entry) > 0:
        print(f"\nLate-entry countries (start > 2005Q4): {list(late_entry.index)}")
        print("These will be handled by cold-start logic in 1.3.")


def main():
    print("=" * 60)
    print("Phase 1.1 — Quarterly data pull")
    print("=" * 60)

    # --- QNA level ---
    print(f"\n[1/5] Reading QNA level from {QNA_LEVEL_FILE.name} ...")
    level = pull_qna_level(QNA_LEVEL_FILE)
    print(f"  {len(level)} rows, {level['country_code'].nunique()} countries")

    # --- Compute growth from level ---
    print("\n[2/5] Computing Q/Q and Y/Y growth from level ...")
    level = compute_growth_from_level(level)

    # --- QNA growth ---
    print(f"\n[3/5] Reading QNA growth from {QNA_GROWTH_FILE.name} ...")
    growth = pull_qna_growth(QNA_GROWTH_FILE)
    print(f"  {len(growth)} rows, {growth['country_code'].nunique()} countries")

    # --- Merge and reconcile ---
    print("\n[4/5] Merging level + published growth and reconciling ...")
    panel = level.merge(growth, on=["country_code", "year_quarter"], how="left")
    panel = reconcile_growth(panel)
    print(f"  Reconciliation: {log_reconciliation_summary(panel)}")

    # --- CLI ---
    print(f"\n[5/5] Reading CLI from {CLI_FILE.name} ...")
    cli = pull_cli(CLI_FILE)
    print(f"  {len(cli)} rows, {cli['country_code'].nunique()} countries")

    # --- STES (optional) ---
    if STES_FILE.exists():
        print(f"\n[STES] Reading STES from {STES_FILE.name} ...")
        stes = pull_stes(STES_FILE)
        print(f"  {len(stes)} rows, {stes['country_code'].nunique()} countries")
    else:
        print("\n[STES] File not yet downloaded — skipping.")

    # --- Output ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved panel to {OUTPUT_PATH}")
    print(f"  Columns: {list(panel.columns)}")
    print(f"  Shape: {panel.shape}")

    # --- Save CLI separately (monthly, will be merged in feature engineering) ---
    cli_path = DATA_DIR / "cli_monthly.parquet"
    cli.to_parquet(cli_path, index=False)
    print(f"Saved CLI to {cli_path}")

    # --- Coverage log ---
    log_coverage(panel, COVERAGE_PATH)
    print(f"Coverage log written to {COVERAGE_PATH}")

    if dropped := TARGET_COUNTRIES - set(panel["country_code"].unique()):
        print(f"Countries dropped (no chain-linked quarterly level): {sorted(dropped)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
