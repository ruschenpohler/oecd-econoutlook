"""
labels.py — Canonical variable labels and figure annotation helpers.

Import in any script or notebook cell:
    import sys, os; sys.path.insert(0, os.path.abspath('../src'))
    from labels import SHORT, DEFS, add_footer
"""

import textwrap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data source line (appears on every figure, distinct from variable notes)
# ---------------------------------------------------------------------------
SOURCE = "Source: OECD Economic Outlook via SDMX REST API (dataflow OECD.ECO.MAD,DSD_EO@DF_EO), 38 member countries, 1990–2027."

# ---------------------------------------------------------------------------
# Short labels: axis ticks, legend entries, chart titles.
# ---------------------------------------------------------------------------
SHORT = {
    "gdpv_annpct":       "GDP growth",
    "unr":               "Unemployment",
    "cbgdpr":            "Current account",
    "itv_annpct":        "Investment growth",
    "xgsv_annpct":       "Export growth",
    "mgsv_annpct":       "Import growth",
    # lag versions
    "gdp_lag1":          "GDP growth (t−1)",
    "gdp_lag2":          "GDP growth (t−2)",
    "gdp_accel":         "GDP acceleration",
    "unr_lag1":          "Unemployment (t−1)",
    "unr_lag2":          "Unemployment (t−2)",
    "cbgdpr_lag1":       "Current account (t−1)",
    "cbgdpr_lag2":       "Current account (t−2)",
    "itv_annpct_lag1":   "Investment growth (t−1)",
    "itv_annpct_lag2":   "Investment growth (t−2)",
    "xgsv_annpct_lag1":  "Export growth (t−1)",
    "xgsv_annpct_lag2":  "Export growth (t−2)",
    "mgsv_annpct_lag1":  "Import growth (t−1)",
    "mgsv_annpct_lag2":  "Import growth (t−2)",
    "country_idx":       "Country FE",
}

# ---------------------------------------------------------------------------
# Variable definitions: (short_label, OECD_code, definition)
# ---------------------------------------------------------------------------
DEFS = {
    "gdpv_annpct": ("GDP growth",        "GDPV_ANNPCT", "Real GDP growth, annual % change"),
    "unr":         ("Unemployment",      "UNR",         "Unemployment rate, % of labour force (ILO-harmonised)"),
    "cbgdpr":      ("Current account",   "CBGDPR",      "Current account balance, % of GDP"),
    "itv_annpct":  ("Investment growth", "ITV_ANNPCT",  "Gross fixed capital formation, volume, annual % change"),
    "xgsv_annpct": ("Export growth",     "XGSV_ANNPCT", "Export volume, annual % change"),
    "mgsv_annpct": ("Import growth",     "MGSV_ANNPCT", "Import volume, annual % change"),
}


def add_footer(fig, vars_used: list, extra_notes: str = None,
               fontsize: int = 7, y_notes: float = 0.025, y_source: float = 0.008):
    """
    Add two footer lines to a figure:
      Line 1 (Notes):  variable definitions + any extra_notes string
      Line 2 (Source): standard data source citation

    Parameters
    ----------
    fig         : matplotlib Figure
    vars_used   : list of variable keys (from DEFS) to document
    extra_notes : optional free-text appended after variable definitions
    fontsize    : font size for both lines
    y_notes     : vertical position of Notes line in figure coordinates
    y_source    : vertical position of Source line in figure coordinates
    """
    # Build Notes line
    parts = []
    for v in vars_used:
        if v in DEFS:
            short, code, defn = DEFS[v]
            parts.append(f"{short} ({code}): {defn}")
    notes_text = "Notes: " + "; ".join(parts) + "."
    if extra_notes:
        notes_text = notes_text.rstrip(".") + ". " + extra_notes

    wrapped_notes = "\n".join(textwrap.wrap(notes_text, width=130))

    fig.text(0.01, y_notes, wrapped_notes,
             ha="left", va="top", fontsize=fontsize,
             color="#444444", style="italic",
             transform=fig.transFigure)

    fig.text(0.01, y_source, SOURCE,
             ha="left", va="top", fontsize=fontsize,
             color="#888888",
             transform=fig.transFigure)
