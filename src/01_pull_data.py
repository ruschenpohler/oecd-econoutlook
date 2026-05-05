"""
Phase 1 data pull: QNA (GDP level + components + growth) + KEI (UNEMP, CA_GDP) + CLI.

Reads from locally downloaded OECD Data Explorer CSV exports.
One entry point, one output file.

Usage:
    uv run python src/01_pull_data.py
"""

from pathlib import Path

import pandas as pd

from data_quality import compute_growth_from_level, log_reconciliation_summary, reconcile_growth

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
QNA_LEVEL_FILE = DATA_DIR / "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,1.1+all.csv"
QNA_GROWTH_FILE = DATA_DIR / "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,+all.csv"
CLI_FILE = DATA_DIR / "OECD.SDD.STES,DSD_STES@DF_CLI,4.1+all.csv"
KEI_FILE = DATA_DIR / "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0+all.csv"

# ---------------------------------------------------------------------------
# 38 OECD member country codes (from v1)
# ---------------------------------------------------------------------------
TARGET_COUNTRIES = {
    "AUS",
    "AUT",
    "BEL",
    "CAN",
    "CHL",
    "COL",
    "CRI",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GBR",
    "GRC",
    "HUN",
    "IRL",
    "ISL",
    "ISR",
    "ITA",
    "JPN",
    "KOR",
    "LTU",
    "LUX",
    "LVA",
    "MEX",
    "NLD",
    "NOR",
    "NZL",
    "POL",
    "PRT",
    "SVK",
    "SVN",
    "SWE",
    "CHE",
    "TUR",
    "USA",
}

# ---------------------------------------------------------------------------
# QNA: GDP level
# ---------------------------------------------------------------------------


def pull_qna_gdp(path: Path) -> pd.DataFrame:
    """Real GDP chain-linked level (B1GQ, L, Y, S1, Q)."""
    df = pd.read_csv(path, low_memory=False)
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
    df["gdp_level_real"] = pd.to_numeric(df["gdp_level_real"], errors="coerce")
    df = df.dropna(subset=["gdp_level_real"])
    df = df[df["country_code"].isin(TARGET_COUNTRIES)].reset_index(drop=True)
    return df


def pull_qna_components(path: Path) -> pd.DataFrame:
    """
    Extract GFCF (P51G), exports (P6), imports (P7) chain-linked quarterly levels.
    Returns wide DataFrame: country_code, year_quarter, itv_level, xgsv_level, mgsv_level
    """
    df = pd.read_csv(path, low_memory=False)
    mask = (
        (df["TRANSACTION"].isin(["P51G", "P6", "P7"]))
        & (df["PRICE_BASE"] == "L")
        & (df["ADJUSTMENT"] == "Y")
        & (df["SECTOR"] == "S1")
        & (df["FREQ"] == "Q")
        & (df["UNIT_MEASURE"] == "XDC")
    )
    df = df.loc[mask, ["REF_AREA", "TIME_PERIOD", "TRANSACTION", "OBS_VALUE"]].copy()
    df.columns = ["country_code", "year_quarter", "transaction", "obs_value"]
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce")

    pivoted = df.pivot_table(
        index=["country_code", "year_quarter"],
        columns="transaction",
        values="obs_value",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"P51G": "itv_level", "P6": "xgsv_level", "P7": "mgsv_level"})
    pivoted = pivoted[pivoted["country_code"].isin(TARGET_COUNTRIES)].reset_index(drop=True)
    return pivoted


# ---------------------------------------------------------------------------
# QNA: published growth (for reconciliation)
# ---------------------------------------------------------------------------


def pull_qna_growth(path: Path) -> pd.DataFrame:
    """OECD-published Q/Q (G1) and Y/Y (GY) GDP growth for reconciliation."""
    df = pd.read_csv(path, low_memory=False)
    mask = (
        (df["TRANSACTION"] == "B1GQ")
        & (df["ADJUSTMENT"] == "Y")
        & (df["TRANSFORMATION"].isin(["G1", "GY"]))
    )
    df = df.loc[mask, ["REF_AREA", "TIME_PERIOD", "TRANSFORMATION", "OBS_VALUE"]].copy()
    df.columns = ["country_code", "year_quarter", "transformation", "obs_value"]
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce")

    pivoted = df.pivot_table(
        index=["country_code", "year_quarter"],
        columns="transformation",
        values="obs_value",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"G1": "gdpv_qq_published", "GY": "gdpv_yy_published"})
    pivoted = pivoted[pivoted["country_code"].isin(TARGET_COUNTRIES)].reset_index(drop=True)
    return pivoted


# ---------------------------------------------------------------------------
# KEI: unemployment and current account
# ---------------------------------------------------------------------------


def pull_kei(path: Path) -> pd.DataFrame:
    """
    Extract unemployment rate (UNEMP, PT_LF, Y, Q) and current account (CA_GDP, PT_B1GQ, Y, Q).
    Returns wide DataFrame: country_code, year_quarter, unr, cbgdpr
    """
    df = pd.read_csv(path, low_memory=False)
    mask = (
        (df["MEASURE"].isin(["UNEMP", "CA_GDP"]))
        & (df["ADJUSTMENT"] == "Y")
        & (df["FREQ"] == "Q")
        & (df["UNIT_MEASURE"].isin(["PT_LF", "PT_B1GQ"]))
    )
    df = df.loc[mask, ["REF_AREA", "TIME_PERIOD", "MEASURE", "OBS_VALUE"]].copy()
    df.columns = ["country_code", "year_quarter", "measure", "obs_value"]
    df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce")

    pivoted = df.pivot_table(
        index=["country_code", "year_quarter"],
        columns="measure",
        values="obs_value",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"UNEMP": "unr", "CA_GDP": "cbgdpr"})
    pivoted = pivoted[pivoted["country_code"].isin(TARGET_COUNTRIES)].reset_index(drop=True)
    return pivoted


# ---------------------------------------------------------------------------
# CLI series
# ---------------------------------------------------------------------------


def pull_cli(path: Path) -> pd.DataFrame:
    """Amplitude-adjusted CLI index (LI + AA + IX), monthly."""
    df = pd.read_csv(path, low_memory=False)
    mask = (df["MEASURE"] == "LI") & (df["ADJUSTMENT"] == "AA") & (df["TRANSFORMATION"] == "IX")
    df = df.loc[mask, ["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["country_code", "year_month", "cli"]
    df["cli"] = pd.to_numeric(df["cli"], errors="coerce")
    df = df.dropna(subset=["cli"])

    available = set(df["country_code"].unique())
    oecd_with_cli = available & TARGET_COUNTRIES
    oecd_without_cli = TARGET_COUNTRIES - available
    print(f"  CLI available for {len(oecd_with_cli)} OECD members: {sorted(oecd_with_cli)}")
    print(f"  CLI not available for {len(oecd_without_cli)} OECD members (no headline LI series)")
    df = df[df["country_code"].isin(available & TARGET_COUNTRIES)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Growth computation for components
# ---------------------------------------------------------------------------


def compute_component_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Q/Q and Y/Y growth for investment (itv), exports (xgsv), imports (mgsv).
    Adds columns: itv_annpct, itv_yy, xgsv_annpct, xgsv_yy, mgsv_annpct, mgsv_yy
    """
    for prefix, level_col in [("itv", "itv_level"), ("xgsv", "xgsv_level"), ("mgsv", "mgsv_level")]:
        if level_col not in df.columns:
            continue
        df = df.sort_values(["country_code", "year_quarter"])
        g = df.groupby("country_code")[level_col]
        ratio = g.shift(0) / g.shift(1)
        df[f"{prefix}_annpct"] = (ratio**4 - 1) * 100
        df[f"{prefix}_yy"] = (g.shift(0) / g.shift(4) - 1) * 100
    return df


# ---------------------------------------------------------------------------
# Coverage log
# ---------------------------------------------------------------------------


def log_coverage(df: pd.DataFrame, path: Path) -> None:
    """Write per-country coverage log."""
    coverage = (
        df.groupby("country_code")["year_quarter"]
        .agg(start="min", end="max", n_obs="count")
        .sort_values("start")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# QNA quarterly coverage by country\n\n")
        dropped = TARGET_COUNTRIES - set(coverage.index)
        if dropped:
            f.write(
                f"**Dropped countries** (missing chain-linked quarterly GDP level): "
                f"{', '.join(sorted(dropped))}\n\n"
            )
        f.write(
            f"Panel: {coverage['start'].min()} to {coverage['end'].max()}, "
            f"N={len(coverage)} countries, {coverage['n_obs'].sum()} observations\n\n"
        )
        f.write("| country_code | start | end | n_obs |\n")
        f.write("|--------------|-------|-----|-------|\n")
        for country, row in coverage.iterrows():
            f.write(f"| {country} | {row['start']} | {row['end']} | {row['n_obs']} |\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Phase 1.1 — Quarterly data pull")
    print("=" * 60)

    # --- QNA GDP level ---
    print(f"\n[1/7] QNA GDP level from {QNA_LEVEL_FILE.name} ...")
    gdp = pull_qna_gdp(QNA_LEVEL_FILE)
    print(f"  {len(gdp)} rows, {gdp['country_code'].nunique()} countries")

    # --- QNA components ---
    print("\n[2/7] QNA components (GFCF, exports, imports) ...")
    components = pull_qna_components(QNA_LEVEL_FILE)
    print(f"  {len(components)} rows, {components['country_code'].nunique()} countries")

    # --- Merge GDP + components, compute growth ---
    print("\n[3/7] Computing Q/Q and Y/Y growth for GDP and components ...")
    panel = gdp.merge(components, on=["country_code", "year_quarter"], how="outer")
    panel = compute_growth_from_level(panel)
    panel = compute_component_growth(panel)

    # --- KEI ---
    print(f"\n[4/7] KEI indicators from {KEI_FILE.name} ...")
    kei = pull_kei(KEI_FILE)
    print(f"  {len(kei)} rows, {kei['country_code'].nunique()} countries")
    print(f"  UNEMP coverage: {kei['unr'].notna().sum()} obs")
    print(f"  CA_GDP coverage: {kei['cbgdpr'].notna().sum()} obs")

    # --- Merge KEI ---
    panel = panel.merge(kei, on=["country_code", "year_quarter"], how="left")

    # --- QNA published growth ---
    print(f"\n[5/7] QNA published growth from {QNA_GROWTH_FILE.name} ...")
    growth = pull_qna_growth(QNA_GROWTH_FILE)
    print(f"  {len(growth)} rows, {growth['country_code'].nunique()} countries")

    # --- Merge and reconcile GDP growth ---
    print("\n[6/7] Reconciling computed vs published GDP growth ...")
    panel = panel.merge(growth, on=["country_code", "year_quarter"], how="left")
    panel = reconcile_growth(panel)
    print(f"  {log_reconciliation_summary(panel)}")

    # --- CLI ---
    print(f"\n[7/7] CLI from {CLI_FILE.name} ...")
    cli = pull_cli(CLI_FILE)
    print(f"  {len(cli)} rows, {cli['country_code'].nunique()} countries")

    # --- Output ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved panel to {OUTPUT_PATH}")
    print(f"  Columns ({len(panel.columns)}): {list(panel.columns)}")
    print(f"  Shape: {panel.shape}")

    cli.to_parquet(DATA_DIR / "cli_monthly.parquet", index=False)
    print("Saved CLI to cli_monthly.parquet")

    log_coverage(panel, COVERAGE_PATH)
    print(f"Coverage log: {COVERAGE_PATH}")

    if dropped := TARGET_COUNTRIES - set(panel["country_code"].unique()):
        print(f"Countries dropped (no chain-linked quarterly level): {sorted(dropped)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
