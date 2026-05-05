"""
Growth-series reconciliation: cross-validate computed growth against
OECD-published growth rates.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def compute_growth_from_level(
    df: pd.DataFrame,
    level_col: str = "gdp_level_real",
    country_col: str = "country_code",
    period_col: str = "year_quarter",
) -> pd.DataFrame:
    """
    Compute Q/Q simple, Q/Q annualised, and Y/Y growth from real GDP level.

    Q/Q simple  : (Yt / Yt-1 - 1) * 100        — matches OECD G1 convention
    Q/Q ann'd   : ((Yt / Yt-1)^4 - 1) * 100    — target variable for modeling
    Y/Y         : (Yt / Yt-4 - 1) * 100         — matches OECD GY convention
    """
    out = df.sort_values([country_col, period_col]).copy()
    g = out.groupby(country_col)[level_col]
    qoq_ratio = g.shift(0) / g.shift(1)
    out["gdpv_qq_simple"] = (qoq_ratio - 1) * 100
    out["gdpv_qq_annualised"] = (qoq_ratio ** 4 - 1) * 100
    out["gdpv_yy_computed"] = (g.shift(0) / g.shift(4) - 1) * 100
    return out


def reconcile_growth(
    df: pd.DataFrame,
    tolerance_pp: float = 0.1,
    log_path: str = "output/growth_reconciliation.md",
) -> pd.DataFrame:
    """
    Compare computed vs OECD-published growth. Warn on deviations > tolerance.

    Expects columns: gdpv_qq_computed, gdpv_qq_published,
                     gdpv_yy_computed, gdpv_yy_published.
    Returns the input df unchanged; side effect is the reconciliation log.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for measure, c, p in [
        ("qq_simple", "gdpv_qq_simple", "gdpv_qq_published"),
        ("yy", "gdpv_yy_computed", "gdpv_yy_published"),
    ]:
        if c not in df.columns or p not in df.columns:
            continue
        diff = (df[c] - df[p]).abs()
        n_total = diff.notna().sum()
        n_breach = (diff > tolerance_pp).sum()
        worst_idx = diff.nlargest(10).index
        worst = df.loc[worst_idx, ["country_code", "year_quarter", c, p]].copy()
        worst["abs_diff_pp"] = diff.loc[worst_idx].round(3)
        rows.append({
            "measure": measure,
            "n_total": int(n_total),
            "n_breach": int(n_breach),
            "share_breach": round(n_breach / n_total, 4) if n_total else np.nan,
            "worst": worst,
        })
        if n_breach > 0:
            share = n_breach / n_total if n_total else 0
            print(
                f"WARNING: {measure.upper()} growth: {n_breach}/{n_total} "
                f"cells exceed {tolerance_pp}pp tolerance ({share:.2%})"
            )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Growth-series reconciliation\n\n")
        for r in rows:
            f.write(f"## {r['measure'].upper()}\n\n")
            f.write(f"- Cells compared: {r['n_total']}\n")
            f.write(f"- Cells above {tolerance_pp}pp tolerance: "
                    f"{r['n_breach']} ({r['share_breach']:.2%})\n\n")
            f.write("### 10 largest deviations\n\n")
            f.write(r["worst"].to_markdown(index=False) + "\n\n")
    return df

def log_reconciliation_summary(panel: pd.DataFrame) -> str:
    """Return a one-line summary of reconciliation results."""
    checks = [
        ("Simple Q/Q vs OECD G1", "gdpv_qq_simple", "gdpv_qq_published"),
        ("Y/Y vs OECD GY", "gdpv_yy_computed", "gdpv_yy_published"),
    ]
    parts = []
    for label, c, p in checks:
        if c not in panel.columns or p not in panel.columns:
            parts.append(f"{label}: N/A")
            continue
        diff = (panel[c] - panel[p]).abs()
        n = diff.notna().sum()
        n_bad = (diff > 0.1).sum()
        max_d = diff.max()
        parts.append(f"{label}: {n_bad}/{n} breach 0.1pp (max {max_d:.3f})")
    return " | ".join(parts)
