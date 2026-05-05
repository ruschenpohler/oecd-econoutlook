"""
01_pull_data.py — Pull OECD Economic Outlook data via SDMX REST API.

Downloads annual macroeconomic indicators for all OECD member countries,
pivots wide (one row per country-year), and saves to data/oecd_economic_outlook.csv.

Data source: OECD Economic Outlook via SDMX
  Dataflow: OECD.ECO.MAD,DSD_EO@DF_EO,
  API docs: https://data-explorer.oecd.org/
"""

import os
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENDPOINT = "https://sdmx.oecd.org/public/rest/data"
DATAFLOW = "OECD.ECO.MAD,DSD_EO@DF_EO,"

# Variables to pull (MEASURE dimension codes).
# Codes verified against DSD_EO@DF_EO available measures (217 total).
# Dropped CPI_YTYPCT (headline inflation) and SRATIO (household saving ratio)
# due to structural missingness:
#   - CPI_YTYPCT: entirely missing for all Eurozone members (reported at
#     aggregate level only in the Economic Outlook)
#   - SRATIO: entirely missing for 26 countries including major OECD members
#     (FRA, GBR, GRC, PRT, ISL, ISR, TUR, ...)
# Remaining 6 features still capture demand (investment, trade), external
# balance (current account), and labour market slack (unemployment).
MEASURES = {
    "GDPV_ANNPCT":  "Real GDP growth (%)",               # ← target variable
    "UNR":          "Unemployment rate (%)",
    "CBGDPR":       "Current account balance (% GDP)",
    "ITV_ANNPCT":   "Gross fixed capital formation, volume, growth (%)",
    "XGSV_ANNPCT":  "Export volume growth (%)",
    "MGSV_ANNPCT":  "Import volume growth (%)",
}

# Positive list: OECD member country codes (as of 2024, 38 members).
# We restrict to actual member states to avoid aggregates (OECD, EA20, G7, W, ...),
# non-OECD partner economies (ARG, BRA, CHN, IDN, IND, ...), and commodity
# groupings (OIL_O, OIL_SAU_O). This keeps the empirical question clean:
# "nowcasting GDP growth across OECD member economies."
OECD_MEMBERS = {
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DEU", "DNK",
    "ESP", "EST", "FIN", "FRA", "GBR", "GRC", "HUN", "IRL", "ISL", "ISR",
    "ITA", "JPN", "KOR", "LTU", "LUX", "LVA", "MEX", "NLD", "NOR", "NZL",
    "POL", "PRT", "SVK", "SVN", "SWE", "CHE", "TUR", "USA",
}

# Paths — resolve relative to project root so the script works from any cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "oecd_economic_outlook.csv")

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def build_url():
    """Construct the SDMX REST query URL.

    URL pattern:
      {endpoint}/{dataflow}/{key}?params
    Key structure for DF_EO (from OECD Data Explorer):
      {REF_AREA}.{MEASURE}.{FREQUENCY}
    We leave REF_AREA empty (= all countries) and pass measures as +-separated.
    """
    measures_str = "+".join(MEASURES.keys())
    # Key: .{MEASURES}.A (empty REF_AREA = all, A = annual)
    key = f".{measures_str}.A"
    params = (
        "startPeriod=1990"
        "&dimensionAtObservation=AllDimensions"
        "&format=csvfilewithlabels"
    )
    return f"{ENDPOINT}/{DATAFLOW}/{key}?{params}"


def download_data():
    """Fetch data from the OECD SDMX API and return a raw pandas DataFrame."""
    url = build_url()
    print(f"Fetching data from:\n  {url}\n")

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    # The API returns CSV; read it directly from the response text
    from io import StringIO
    raw = pd.read_csv(StringIO(resp.text))
    print(f"Raw download: {raw.shape[0]:,} rows × {raw.shape[1]} columns")
    return raw


# ---------------------------------------------------------------------------
# Clean & pivot
# ---------------------------------------------------------------------------

def clean_and_pivot(raw: pd.DataFrame) -> pd.DataFrame:
    """Transform the long SDMX CSV into a wide panel dataset.

    The raw CSV has one row per observation with columns like:
      REF_AREA, Reference area, MEASURE, Measure, TIME_PERIOD, OBS_VALUE, ...
    We pivot to: one row per (country_code, country_name, year), one column per variable.
    """
    # Identify the columns we need — SDMX CSV column names can vary slightly,
    # so we handle both labelled and code-only formats
    col_map = {}
    for candidate, target in [
        ("REF_AREA", "country_code"),
        ("Reference area", "country_name"),
        ("MEASURE", "measure"),
        ("TIME_PERIOD", "year"),
        ("OBS_VALUE", "value"),
    ]:
        if candidate in raw.columns:
            col_map[candidate] = target

    df = raw.rename(columns=col_map)

    # If country_name column wasn't in the CSV, create a placeholder
    if "country_name" not in df.columns:
        df["country_name"] = df["country_code"]

    # Keep only the columns we need
    df = df[["country_code", "country_name", "measure", "year", "value"]].copy()

    # Convert types
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Keep only OECD member countries (positive list).
    # Drops aggregates (OECD, EA20, G7, W, ...), non-OECD partners (ARG, BRA,
    # CHN, ...), and commodity groupings (OIL_O, OIL_SAU_O).
    all_codes = set(df["country_code"].unique())
    n_before = len(all_codes)
    dropped = sorted(all_codes - OECD_MEMBERS)
    df = df[df["country_code"].isin(OECD_MEMBERS)]
    n_after = df["country_code"].nunique()
    print(f"Kept {n_after} OECD members, dropped {n_before - n_after} "
          f"non-member/aggregate entities: {', '.join(dropped)}")

    # Pivot: one row per (country, year), one column per measure
    df_wide = df.pivot_table(
        index=["country_code", "country_name", "year"],
        columns="measure",
        values="value",
        aggfunc="first",  # should be one value per cell; first handles any duplicates
    ).reset_index()

    # Flatten MultiIndex columns from pivot
    df_wide.columns.name = None

    # Lowercase column names for consistency
    df_wide.columns = [c.lower() for c in df_wide.columns]

    # Sort for readability
    df_wide = df_wide.sort_values(["country_code", "year"]).reset_index(drop=True)

    return df_wide


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    """Print a concise summary of the cleaned dataset."""
    countries = sorted(df["country_code"].unique())
    year_min = df["year"].min()
    year_max = df["year"].max()

    print(f"\n{'='*60}")
    print(f"OECD Economic Outlook — Cleaned Dataset")
    print(f"{'='*60}")
    print(f"Shape:      {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Countries:  {len(countries)} ({countries[0]} ... {countries[-1]})")
    print(f"Years:      {year_min}–{year_max}")
    print(f"\nVariables:")
    for code, desc in MEASURES.items():
        col = code.lower()
        if col in df.columns:
            n_miss = df[col].isna().sum()
            pct_miss = 100 * n_miss / len(df)
            print(f"  {col:<20s} {desc:<35s} missing: {n_miss:>4d} ({pct_miss:.1f}%)")
        else:
            print(f"  {col:<20s} {desc:<35s} ** NOT FOUND IN DATA **")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    raw = download_data()
    df = clean_and_pivot(raw)
    print_summary(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
