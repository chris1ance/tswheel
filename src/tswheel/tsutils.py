"""Time series utilities."""

import pandas as pd
from arch.unitroot import ADF, PhillipsPerron, ZivotAndrews, KPSS, VarianceRatio
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


def check_stationarity(
    y: pd.Series,
    lags: int = 1,
) -> pd.DataFrame:
    """
    Performs a battery of unit root tests (ADF, Phillips-Perron, Zivot-Andrews,
    and KPSS) on a time series.

    Runs the following tests:
    1. ADF with trend='n'
    2. ADF with trend='c'
    3. Phillips-Perron with trend='n' and test_type='tau'
    4. Phillips-Perron with trend='n' and test_type='rho'
    5. Phillips-Perron with trend='c' and test_type='tau'
    6. Phillips-Perron with trend='c' and test_type='rho'
    7. Zivot-Andrews with trend='c'
    8. KPSS with trend='c'

    Note: The null hypothesis for ADF, Phillips-Perron, and Zivot-Andrews
    is a unit root (non-stationarity), while the null for KPSS is
    stationarity.

    Args:
        y (Union[pd.Series, np.ndarray]): The time series data to test.
        lags (int | None, default=1): The number of lags to use in the ADF,
                                       Zivot-Andrews regressions.

    Returns:
        pd.DataFrame: A DataFrame containing the results of each test,
                      including Test name, Trend, Test Type (for PP), Statistic,
                      P-value, and Lags used.

    Raises:
        TypeError: If y is not a pandas Series or numpy ndarray.
        ValueError: If an invalid method is provided for lag selection.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from arch.unitroot import ADF # Required for example
        >>> np.random.seed(0)
        >>> random_walk = np.random.randn(1000).cumsum()
        >>> series = pd.Series(random_walk)
        >>> results = check_stationarity(series)
        >>> print(results)
    """
    # --- Input Validation ---
    if not isinstance(y, pd.Series):
        raise TypeError("Input `y` must be a pandas Series.")

    # --- Perform Tests ---
    results = []

    # ADF tests
    adf_n = ADF(y, trend="n", lags=lags)
    results.append(
        {
            "Test": "ADF",
            "Trend": "n",
            "Test Type": None,
            "Statistic": adf_n.stat,
            "P-value": adf_n.pvalue,
            "Lags": adf_n.lags,
            "HAC Lags": None,
        }
    )

    adf_c = ADF(y, trend="c", lags=lags)
    results.append(
        {
            "Test": "ADF",
            "Trend": "c",
            "Test Type": None,
            "Statistic": adf_c.stat,
            "P-value": adf_c.pvalue,
            "Lags": adf_c.lags,
            "HAC Lags": None,
        }
    )

    # Phillips-Perron tests
    pp_n_tau = PhillipsPerron(y, trend="n", test_type="tau")
    results.append(
        {
            "Test": "Phillips-Perron",
            "Trend": "n",
            "Test Type": "tau",
            "Statistic": pp_n_tau.stat,
            "P-value": pp_n_tau.pvalue,
            "Lags": None,
            "HAC Lags": pp_n_tau.lags,
        }
    )

    pp_n_rho = PhillipsPerron(y, trend="n", test_type="rho")
    results.append(
        {
            "Test": "Phillips-Perron",
            "Trend": "n",
            "Test Type": "rho",
            "Statistic": pp_n_rho.stat,
            "P-value": pp_n_rho.pvalue,
            "Lags": None,
            "HAC Lags": pp_n_rho.lags,
        }
    )

    pp_c_tau = PhillipsPerron(y, trend="c", test_type="tau")
    results.append(
        {
            "Test": "Phillips-Perron",
            "Trend": "c",
            "Test Type": "tau",
            "Statistic": pp_c_tau.stat,
            "P-value": pp_c_tau.pvalue,
            "Lags": None,
            "HAC Lags": pp_c_tau.lags,
        }
    )

    pp_c_rho = PhillipsPerron(y, trend="c", test_type="rho")
    results.append(
        {
            "Test": "Phillips-Perron",
            "Trend": "c",
            "Test Type": "rho",
            "Statistic": pp_c_rho.stat,
            "P-value": pp_c_rho.pvalue,
            "Lags": None,
            "HAC Lags": pp_c_rho.lags,
        }
    )

    # Zivot-Andrews test
    za_c = ZivotAndrews(y, trend="c", lags=lags)
    results.append(
        {
            "Test": "Zivot-Andrews",
            "Trend": "c",
            "Test Type": None,  # Zivot-Andrews doesn't have a test_type like PP
            "Statistic": za_c.stat,
            "P-value": za_c.pvalue,
            "Lags": za_c.lags,
            "HAC Lags": None,
        }
    )

    # KPSS test
    kpss_c = KPSS(y, trend="c", lags=lags)
    results.append(
        {
            "Test": "KPSS",
            "Trend": "c",
            "Test Type": None,  # KPSS doesn't have a test_type like PP
            "Statistic": kpss_c.stat,
            "P-value": 1 - kpss_c.pvalue,
            "Lags": None,
            "HAC Lags": kpss_c.lags,
        }
    )

    # Variance Ratio tests
    vr_n = VarianceRatio(y, trend="n", lags=2)
    results.append(
        {
            "Test": "VarianceRatio",
            "Trend": "n",
            "Test Type": None,
            "Statistic": vr_n.stat,
            "P-value": vr_n.pvalue,
            "Lags": vr_n.lags,
            "HAC Lags": None,
        }
    )

    vr_c = VarianceRatio(y, trend="c", lags=2)
    results.append(
        {
            "Test": "VarianceRatio",
            "Trend": "c",
            "Test Type": None,
            "Statistic": vr_c.stat,
            "P-value": vr_c.pvalue,
            "Lags": vr_c.lags,
            "HAC Lags": None,
        }
    )

    # --- Format Results ---
    results_df = pd.DataFrame(results)

    return results_df
