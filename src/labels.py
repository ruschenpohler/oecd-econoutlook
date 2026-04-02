"""
labels.py — Canonical variable labels and figure annotation helpers.

Import in any script or notebook cell:
    from labels import SHORT, NOTES_LINE, add_notes
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

# ---------------------------------------------------------------------------
# Short labels: what appears on axis ticks, legend entries, chart titles.
# ---------------------------------------------------------------------------
SHORT = {
    "gdpv_annpct":  "GDP growth",
    "unr":          "Unemployment",
    "cbgdpr":       "Current account",
    "itv_annpct":   "Investment growth",
    "xgsv_annpct":  "Export growth",
    "mgsv_annpct":  "Import growth",
    # lag versions — used in Phase 2/3 plots
    "gdp_lag1":     "GDP growth (t−1)",
    "gdp_lag2":     "GDP growth (t−2)",
    "gdp_accel":    "GDP acceleration",
    "unr_lag1":     "Unemployment (t−1)",
    "unr_lag2":     "Unemployment (t−2)",
    "cbgdpr_lag1":  "Current account (t−1)",
    "cbgdpr_lag2":  "Current account (t−2)",
    "itv_annpct_lag1": "Investment growth (t−1)",
    "itv_annpct_lag2": "Investment growth (t−2)",
    "xgsv_annpct_lag1": "Export growth (t−1)",
    "xgsv_annpct_lag2": "Export growth (t−2)",
    "mgsv_annpct_lag1": "Import growth (t−1)",
    "mgsv_annpct_lag2": "Import growth (t−2)",
    "country_idx":  "Country FE",
}

# ---------------------------------------------------------------------------
# Full definitions: what appears in the Notes line.
# Format: (short_label, dataset_varname, definition)
# ---------------------------------------------------------------------------
DEFS = {
    "gdpv_annpct":  ("GDP growth",       "GDPV_ANNPCT", "Real GDP growth, annual % change"),
    "unr":          ("Unemployment",     "UNR",         "Unemployment rate, % of labour force (ILO-harmonised)"),
    "cbgdpr":       ("Current account",  "CBGDPR",      "Current account balance, % of GDP"),
    "itv_annpct":   ("Investment growth","ITV_ANNPCT",  "Gross fixed capital formation, volume, annual % change"),
    "xgsv_annpct":  ("Export growth",    "XGSV_ANNPCT", "Export volume, annual % change"),
    "mgsv_annpct":  ("Import growth",    "MGSV_ANNPCT", "Import volume, annual % change"),
}


def notes_line(vars_used: list[str], prefix: str = "Notes: ") -> str:
    """
    Build a plain-text notes string for the listed variable keys.
    Returns e.g.:
      Notes: GDP growth (GDPV_ANNPCT): Real GDP growth, annual % change;
             Unemployment (UNR): Unemployment rate, % of labour force (ILO-harmonised); ...
    """
    parts = []
    for v in vars_used:
        if v in DEFS:
            short, code, defn = DEFS[v]
            parts.append(f"{short} ({code}): {defn}")
    return prefix + "; ".join(parts) + "."


def add_notes(fig, vars_used: list[str], fontsize: int = 7, y: float = -0.03):
    """
    Add a Notes: line as a figure-level text annotation below the axes.
    Automatically word-wraps at ~120 chars.

    Usage:
        fig, ax = plt.subplots(...)
        # ... your plot code ...
        add_notes(fig, ["gdpv_annpct", "unr"])
        plt.savefig(...)
    """
    text = notes_line(vars_used)
    wrapped = "\n".join(textwrap.wrap(text, width=120))

    # Build the rich-text version with bold short labels
    # matplotlib's fig.text doesn't support inline bold in plain text,
    # so we write it as a single grey text block using a monospace-adjacent
    # font size. For proper bold inline we'd need a custom renderer.
    fig.text(
        0.01, y, wrapped,
        ha="left", va="top",
        fontsize=fontsize,
        color="#444444",
        style="italic",
        wrap=True,
        transform=fig.transFigure,
    )
