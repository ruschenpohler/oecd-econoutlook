"""
diagnose_truncation.py — After applying OECD-member filter and dropping
cpi_ytypct/sratio, assess how many complete rows we get at each possible
start-year truncation.

Run AFTER re-running 01_pull_data.py with the updated config.
Usage: uv run python src/diagnose_truncation.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "oecd_economic_outlook.csv")

FEATURES = ["gdpv_annpct", "unr", "cbgdpr", "itv_annpct",
            "xgsv_annpct", "mgsv_annpct"]


def main():
    df = pd.read_csv(DATA_PATH)
    countries = sorted(df["country_code"].unique())
    years = sorted(df["year"].unique())
    print(f"Panel after cleaning: {len(countries)} countries × {len(years)} years "
          f"({min(years)}–{max(years)}), {len(df)} rows\n")

    # -----------------------------------------------------------------------
    # 1. Per-country: first year with ALL 6 features non-null
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("FIRST COMPLETE YEAR PER COUNTRY (all 6 features non-null)")
    print("=" * 70)

    first_complete = {}
    for country in countries:
        cdf = df[df["country_code"] == country].sort_values("year")
        complete = cdf[cdf[FEATURES].notna().all(axis=1)]
        if len(complete) > 0:
            first_yr = int(complete["year"].min())
            n_complete = len(complete)
        else:
            first_yr = None
            n_complete = 0
        first_complete[country] = first_yr
        status = f"from {first_yr} ({n_complete} rows)" if first_yr else "NO COMPLETE ROWS"
        print(f"  {country}: {status}")

    # -----------------------------------------------------------------------
    # 2. Truncation analysis: for each candidate start year, how many
    #    countries and complete rows do we retain?
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRUNCATION ANALYSIS: Complete rows by start year")
    print("=" * 70)
    print(f"  {'Start':<8} {'Countries':<12} {'Complete rows':<16} {'% of truncated panel'}")
    print(f"  {'-----':<8} {'---------':<12} {'-------------':<16} {'-------------------'}")

    for start_year in range(1990, 2006):
        truncated = df[df["year"] >= start_year]
        complete = truncated[truncated[FEATURES].notna().all(axis=1)]
        n_countries = complete["country_code"].nunique()
        n_rows = len(complete)
        pct = 100 * n_rows / len(truncated) if len(truncated) > 0 else 0
        marker = " ◄" if start_year in (1995, 2000) else ""
        print(f"  {start_year:<8} {n_countries:<12} {n_rows:<16} {pct:.1f}%{marker}")

    # -----------------------------------------------------------------------
    # 3. Countries that would be lost at each truncation point
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COUNTRIES WITHOUT ANY COMPLETE ROW (would be lost entirely)")
    print("=" * 70)

    for start_year in [1990, 1995, 2000]:
        truncated = df[df["year"] >= start_year]
        complete = truncated[truncated[FEATURES].notna().all(axis=1)]
        present = set(complete["country_code"].unique())
        missing = sorted(set(countries) - present)
        print(f"\n  Start {start_year}: {len(missing)} countries lost: "
              f"{', '.join(missing) if missing else 'none'}")

    # -----------------------------------------------------------------------
    # 4. Variable-level missingness in OECD-member panel
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VARIABLE MISSINGNESS (OECD members only)")
    print("=" * 70)
    for var in FEATURES:
        n_miss = df[var].isna().sum()
        pct = 100 * n_miss / len(df)
        print(f"  {var:<20s} missing: {n_miss:>4d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
