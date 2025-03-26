"""Utilities for FRED data."""

from fredapi import Fred
import pandas as pd
from .datawork.strchecks import is_valid_date_format

pd.set_option("mode.copy_on_write", True)


def get_fred_series(
    api_key: str,
    series: dict[str, str],
    start_date: str | None,
    end_date: str | None = None,
    period_index: bool = False,
) -> pd.DataFrame:
    """
    Pull multiple FRED series and combine them into a single DataFrame with custom column names.

    Parameters:
    api_key (str): Your FRED API key
    series (dict[str, str]): Dictionary mapping FRED series IDs to display names
    start_date (str, optional): Start date in 'YYYY-MM-DD' format
    end_date (str, optional): End date in 'YYYY-MM-DD' format
    period_index (bool, optional): If True, return DataFrame with a Period Index (default: False, for a Datetime Index)

    Returns:
    pandas.DataFrame: Combined DataFrame with series as columns

    Example:
    >>> api_key = "your_api_key_here"
    >>> series = {
    ...     'UNRATE': 'Unemployment Rate',
    ...     'CPIAUCSL': 'Consumer Price Index'
    ... }
    >>> df = get_fred_series(
    ...     api_key=api_key,
    ...     series=series,
    ...     start_date='2000-01-01',
    ...     end_date='2023-12-31'
    ... )
    """

    if start_date:
        is_valid_date_format(start_date)
    if end_date:
        is_valid_date_format(end_date)

    # Initialize FRED
    fred = Fred(api_key=api_key)

    # Get series info to check frequencies
    frequencies = {}
    for series_id in series:
        info = fred.get_series_info(series_id)
        frequencies[series_id] = info["frequency"]

    # Check if all frequencies match
    if len(set(frequencies.values())) > 1:
        raise ValueError(f"Series have different frequencies: {frequencies}")

    # Dictionary to store series
    series_dict = {}

    # Fetch each series
    for series_id, series_name in series.items():
        data = fred.get_series(
            series_id, observation_start=start_date, observation_end=end_date
        )
        series_dict[series_name] = data

    # Combine all series into a DataFrame
    df = pd.DataFrame(series_dict)

    if period_index:
        # Get the frequency for Period conversion
        freq = next(iter(frequencies.values()))

        freq_map = {
            "Monthly": "M",
            "Quarterly": "Q",
            "Annual": "Y",
            "Weekly": "W",
            "Daily": "D",
        }

        period_freq = freq_map.get(freq)

        if not period_freq:
            raise ValueError(f"Unsupported frequency: {freq}")

        # Convert datetime index to Period index
        df.index = pd.PeriodIndex(df.index, freq=period_freq)

    return df
