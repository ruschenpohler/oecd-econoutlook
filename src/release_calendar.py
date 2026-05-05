"""
Stylised release calendar for pseudo-real-time nowcasting.

Each indicator has a publication-lag specified in days from the end of its
reference period. The mask_for_vintage function returns a boolean mask
indicating which (country, period, indicator) cells are observable at a
given vintage date.
"""
from datetime import date, timedelta
from typing import Optional
import pandas as pd

PUBLICATION_LAG_DAYS = {
    "gdpv_qq":     60,
    "gdpv_yy":     60,
    "cli":         30,
    "unr":         60,
    "cbgdpr":      90,
    "itv_annpct":  90,
    "xgsv_annpct": 90,
    "mgsv_annpct": 90,
}


def end_of_period(period: pd.Period) -> date:
    """Last calendar day of the reference period."""
    return period.end_time.date()


def is_observable(indicator: str, period: pd.Period, vintage: date) -> bool:
    """True iff indicator for `period` is published by `vintage`."""
    publication_date = end_of_period(period) + timedelta(
        days=PUBLICATION_LAG_DAYS[indicator]
    )
    return publication_date <= vintage


def mask_for_vintage(
    df: pd.DataFrame,
    vintage: date,
    period_col: str = "year_quarter",
    indicator_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return df with cells set to NaN where (indicator, period) is not yet
    published as of `vintage`.

    `df` is in long format with columns:
        country_code, year_quarter, gdpv_qq, gdpv_yy, cli, unr, ...
    """
    out = df.copy()
    indicators = indicator_cols or [c for c in PUBLICATION_LAG_DAYS if c in df.columns]
    for ind in indicators:
        observable = out[period_col].apply(
            lambda p: is_observable(ind, p, vintage)
        )
        out.loc[~observable, ind] = float("nan")
    return out
