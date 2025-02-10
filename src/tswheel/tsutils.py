"""Time series utilities."""

import pandas as pd
from typing import Literal

pd.set_option("mode.copy_on_write", True)


def annualized_pct_change(
    series: pd.Series,
    n: int,
    freq: Literal["D", "W", "M", "Q", "Y"] | None = None,
) -> pd.Series:
    """
    Calculates the annualized percentage change over n periods for a time series.

    This function takes a pandas Series with either a DatetimeIndex or PeriodIndex and computes
    the annualized rate of change over a specified number of periods. The result is expressed as
    a percentage. The calculation uses the formula:
        100*((series_t/series_{t-n})^(periods_per_year/n) - 1)

    Parameters
    ----------
    series : pd.Series
        Time series data with either a DatetimeIndex or PeriodIndex.
    n : int
        Number of periods to calculate the change over.
    freq : Literal['D','W','M','Q','Y'] | None, optional
        Frequency of the time series. Required if series has DatetimeIndex.
        Valid values are:
        - 'D': Daily
        - 'W': Weekly
        - 'M': Monthly
        - 'Q': Quarterly
        - 'Y': Yearly
        If series has PeriodIndex, freq is automatically determined from the index.

    Returns
    -------
    pd.Series
        Series containing the annualized percentage changes.
        Values are expressed as percentages (e.g., 5.0 means 5%).

    Raises
    ------
    Exception
        If series doesn't have DatetimeIndex or PeriodIndex
        If freq is not provided for series with DatetimeIndex

    Examples
    --------
    >>> # Monthly data with DatetimeIndex
    >>> dates = pd.date_range('2020-01-01', '2021-12-31', freq='M')
    >>> values = pd.Series([100, 102, 104, 106], index=dates)
    >>> annualized_pct_change(values, n=12, freq='M')

    >>> # Quarterly data with PeriodIndex
    >>> periods = pd.period_range('2020Q1', '2021Q4', freq='Q')
    >>> values = pd.Series([100, 105, 110, 115], index=periods)
    >>> annualized_pct_change(values, n=1)
    """

    if n <= 0 or not isinstance(n, int):
        raise Exception("n must be an integer greater than 0.")

    if isinstance(series.index, pd.DatetimeIndex):
        if not freq:
            raise Exception("If series has a DatetimeIndex, must specify freq.")
    elif isinstance(series.index, pd.PeriodIndex):
        freq = series.index.freqstr
    else:
        raise Exception("series must have a DatetimeIndex or a PeriodIndex.")

    annual_periods_map = {
        "M": 12,
        "Q": 4,
        "Q-DEC": 4,
        "Y": 1,
        "Y-DEC": 1,
        "W": 52,
        "D": 365,
    }

    annual_periods = annual_periods_map[freq]

    transformed_series = (
        series.copy()
        .pct_change(
            periods=n, fill_method=None
        )  # = (series_t/series_{t-n})^(periods_per_year/n) - 1
        .add(1)
        .pow(annual_periods / n)
        .sub(1)
        .mul(100)
    )

    return transformed_series
