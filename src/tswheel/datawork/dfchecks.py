import pandas as pd
from typing import Literal
from pandas.api.types import is_numeric_dtype

pd.set_option("mode.copy_on_write", True)


def have_same_index_type(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> tuple[bool, str]:
    """
    Checks whether two pandas DataFrames have the same type of index and optionally the same frequency.

    Args:
        df1 (pd.DataFrame): The first DataFrame to check
        df2 (pd.DataFrame): The second DataFrame to check

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if indices match according to criteria, False otherwise
            - str: Explanation message about the comparison result

    Examples:
        >>> import pandas as pd
        >>> # Same DatetimeIndex type and frequency
        >>> df1 = pd.DataFrame(index=pd.date_range('2022-01-01', periods=3, freq='D'))
        >>> df2 = pd.DataFrame(index=pd.date_range('2022-02-01', periods=3, freq='D'))
        >>> have_same_index_type(df1, df2)
        (True, 'Indices have the same type and frequency')

        >>> # Same DatetimeIndex type but different frequency
        >>> df3 = pd.DataFrame(index=pd.date_range('2022-01-01', periods=3, freq='M'))
        >>> have_same_index_type(df1, df3)
        (False, 'Indices have the same type but different frequencies: <Day> vs <MonthEnd>')

        >>> # Different index types
        >>> df4 = pd.DataFrame(index=pd.RangeIndex(start=0, stop=5))
        >>> have_same_index_type(df1, df4)
        (False, 'Index types differ: DatetimeIndex vs RangeIndex')
    """
    if type(df1.index) is not type(df2.index):
        return (
            False,
            f"Index types differ: {type(df1.index).__name__} vs {type(df2.index).__name__}",
        )

    # Check frequency for time-based indices if requested
    if hasattr(df1.index, "freq") and hasattr(df2.index, "freq"):
        # Handle None frequency case
        freq1 = df1.index.freq
        freq2 = df2.index.freq

        # If both are None, consider them the same
        if freq1 is None and freq2 is None:
            return True, "Indices have the same type but both have undefined frequency"
        # If only one is None, they differ
        elif freq1 is None or freq2 is None:
            return False, "Indices have the same type but one has undefined frequency"
        # If both have frequency, compare them
        elif freq1 != freq2:
            # Use string representation of frequency objects for clarity
            return (
                False,
                f"Indices have the same type but different frequencies: {freq1} vs {freq2}",
            )
        else:
            return True, "Indices have the same type and frequency"

    return True, "Indices have the same type"


def ensure_period_index(
    df: pd.DataFrame, freq: Literal["B", "D", "W", "M", "Q", "Y"] | None = None
) -> pd.DataFrame:
    """
    Ensures the DataFrame index is a pandas PeriodIndex with an allowed frequency.

    Converts the index if it is a DatetimeIndex, a compatible object/string
    index, or contains datetime.date objects. The resulting PeriodIndex must
    have a base frequency belonging to ['B', 'D', 'W', 'M', 'Q', 'Y'].

    - If the index is already a PeriodIndex with an allowed frequency, the
      DataFrame is returned unchanged.
    - If the index is a DatetimeIndex, it's converted using `to_period()`.
    - If the index is of another potentially convertible type (e.g., object dtype
      with period-like strings or datetime.date objects), it attempts conversion
      first to DatetimeIndex using `pd.to_datetime()`, then to PeriodIndex.
    - If the index is a numeric type (checked using `pandas.api.types.is_numeric_dtype`),
      a TypeError is raised as these cannot be reliably converted.

    The optional `freq` argument can be provided to specify the desired
    frequency for conversion, overriding inference. If `freq` is None, the
    function attempts to infer the frequency.

    Args:
        df (pd.DataFrame): The DataFrame to check and potentially convert.
        freq (str | None, optional): The target frequency for the PeriodIndex.
            Allowed values: 'B' (Business Day), 'D' (Calendar Day), 'W' (Weekly),
            'M' (Month), 'Q' (Quarter), 'Y' (Year).
            If None (default), frequency is inferred during conversion.
            If provided, this frequency is used.

    Returns:
        pd.DataFrame: A copy of the DataFrame with a PeriodIndex, or the
                      original DataFrame if it already had a PeriodIndex with an
                      allowed frequency.

    Raises:
        ValueError: If frequency inference fails (e.g., irregular DatetimeIndex)
                    and `freq` is not provided, or if conversion to PeriodIndex
                    fails after intermediate DatetimeIndex conversion.
        TypeError: If the input is not a pandas DataFrame, if the index is a
                   non-convertible numeric type, or if the index cannot be
                   parsed as datetime objects for conversion.
        AssertionError: If the resulting PeriodIndex frequency (either existing,
                        inferred, or specified) does not have a base frequency
                        within the allowed set ['B', 'D', 'W', 'M', 'Q', 'Y'].
                        This typically occurs if `to_period` infers an unsupported
                        frequency (e.g., '2D') and `freq` is not specified.
        RuntimeError: If an unexpected error occurs during conversion.

    Examples:
        >>> import pandas as pd
        >>> from datetime import date
        >>>
        >>> # DataFrame with DatetimeIndex
        >>> dates = pd.date_range('2023-01-01', periods=3, freq='D')
        >>> df_dt = pd.DataFrame({'data': [1, 2, 3]}, index=dates)
        >>> df_period_dt = ensure_period_index(df_dt.copy())
        >>> print(isinstance(df_period_dt.index, pd.PeriodIndex))
        True
        >>> print(df_period_dt.index.freqstr)
        D
        >>>
        >>> # DataFrame with PeriodIndex (no change)
        >>> periods = pd.period_range('2023-01', periods=3, freq='ME') # Use 'ME'
        >>> df_p = pd.DataFrame({'data': [4, 5, 6]}, index=periods)
        >>> df_p_checked = ensure_period_index(df_p)
        >>> print(df_p_checked.index is df_p.index)
        True
        >>>
        >>> # DataFrame with string index (convertible)
        >>> str_idx = ['2023-01', '2023-02', '2023-03']
        >>> df_str = pd.DataFrame({'data': [7, 8, 9]}, index=str_idx)
        >>> df_period_str = ensure_period_index(df_str.copy()) # Infer freq='ME'
        >>> print(isinstance(df_period_str.index, pd.PeriodIndex))
        True
        >>> print(df_period_str.index.freqstr)
        ME
        >>>
        >>> # DataFrame with datetime.date index
        >>> date_idx = [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
        >>> df_date = pd.DataFrame({'data': [10, 11, 12]}, index=date_idx)
        >>> df_period_date = ensure_period_index(df_date.copy(), freq='ME') # Specify freq 'ME'
        >>> print(isinstance(df_period_date.index, pd.PeriodIndex))
        True
        >>> print(df_period_date.index.freqstr)
        ME
        >>>
        >>> # DataFrame with non-convertible numeric index (raises TypeError)
        >>> df_range = pd.DataFrame({'data': [13, 14, 15]}, index=pd.RangeIndex(3))
        >>> try:
        ...     ensure_period_index(df_range)
        ... except TypeError as e:
        ...     print(e) # doctest: +ELLIPSIS
        Index type RangeIndex with dtype int64 cannot be converted to PeriodIndex...
        >>>
        >>> # DataFrame with DatetimeIndex inferring unsupported freq (raises AssertionError)
        >>> irregular_dates = pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-05'])
        >>> df_irreg = pd.DataFrame({'data': [1, 2, 3]}, index=irregular_dates)
        >>> try:
        ...     ensure_period_index(df_irreg.copy()) # Infers '2D', which is not allowed
        ... except AssertionError as e:
        ...     print(e) # doctest: +ELLIPSIS
        Resulting PeriodIndex frequency '2D' (base: '2D') is not in ['B', 'D', 'W', 'M', 'Q', 'Y']
        >>>
        >>> # Force allowed frequency for irregular dates
        >>> df_irreg_period_forced_d = ensure_period_index(df_irreg.copy(), freq='D') # Force 'D'
        >>> print(df_irreg_period_forced_d.index)
        PeriodIndex(['2023-01-01', '2023-01-03', '2023-01-05'], dtype='period[D]')
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Define the allowed base frequencies using preferred aliases
    ALLOWED_BASE_FREQS = ["B", "D", "W", "M", "Q", "Y"]

    # 1. Check for definitely non-convertible numeric index types
    if is_numeric_dtype(df.index):
        raise TypeError(
            f"Index type {type(df.index).__name__} with dtype {df.index.dtype} "
            "cannot be converted to PeriodIndex. Only DatetimeIndex or potentially convertible "
            "object/string/date indexes are supported."
        )

    # 2. Check if already PeriodIndex
    if isinstance(df.index, pd.PeriodIndex):
        # Ensure the existing frequency is valid according to the new rule
        assert df.index.freq in ALLOWED_BASE_FREQS, (
            f"Existing PeriodIndex frequency '{df.index.freqstr}' is not in {ALLOWED_BASE_FREQS}"
        )
        # Return the original DataFrame as no copy is needed
        return df

    # 3. Check if DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            # Use provided freq if available, otherwise attempt inference
            new_index = df.index.to_period(freq=freq)
            # Create a copy to avoid modifying the original DataFrame
            df_copy = df.copy()
            df_copy.index = new_index
            # Assert the resulting frequency is valid
            assert df_copy.index.freq in ALLOWED_BASE_FREQS, (
                f"Resulting PeriodIndex frequency '{df_copy.index.freqstr}' is not in {ALLOWED_BASE_FREQS}"
            )
            return df_copy
        except ValueError as e:
            # Reraise with a more specific message if conversion fails
            raise ValueError(
                "Cannot convert DatetimeIndex to PeriodIndex. Frequency inference "
                f"failed, and no 'freq' was provided. Original error: {e}"
            ) from e
        except Exception as e:
            # Catch other potential errors during conversion
            # Re-raise AssertionError if it was caught, otherwise wrap as RuntimeError
            if isinstance(e, AssertionError):
                raise e
            raise RuntimeError(
                f"An unexpected error occurred during DatetimeIndex conversion: {e}"
            ) from e

    # 4. Handle other potentially convertible types (e.g., object, string, date)
    # This block is reached if it's not numeric, not PeriodIndex, not DatetimeIndex.
    try:
        # Attempt conversion via pd.to_datetime first
        datetime_index = pd.to_datetime(df.index)
        # Now convert the intermediate DatetimeIndex to PeriodIndex
        try:
            new_index = datetime_index.to_period(freq=freq)
            # Create a copy
            df_copy = df.copy()
            df_copy.index = new_index
            # Assert the resulting frequency is valid
            assert df_copy.index.freq in ALLOWED_BASE_FREQS, (
                f"Resulting PeriodIndex frequency '{df_copy.index.freqstr}' is not in {ALLOWED_BASE_FREQS}"
            )
            return df_copy
        except ValueError as e:
            # Handle failure in to_period (e.g., freq inference failed on the intermediate DatetimeIndex)
            raise ValueError(
                "Could not convert to PeriodIndex after converting index to DatetimeIndex. "
                f"Frequency inference may have failed. Original error: {e}"
            ) from e
        except Exception as e:
            # Re-raise AssertionError if it was caught, otherwise wrap as RuntimeError
            if isinstance(e, AssertionError):
                raise e
            raise RuntimeError(
                f"An unexpected error occurred during PeriodIndex conversion: {e}"
            ) from e

    except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
        # Catch errors indicating the original index wasn't convertible to DatetimeIndex
        raise TypeError(
            f"Index type {type(df.index).__name__} cannot be converted to PeriodIndex. "
            f"Could not parse index values as datetime objects. Original error: {e}"
        ) from e
    except Exception as e:
        # Catch other unexpected errors during the initial to_datetime conversion
        # Re-raise AssertionError if it was caught, otherwise wrap as RuntimeError
        if isinstance(e, AssertionError):
            raise e
        raise RuntimeError(
            f"An unexpected error occurred during initial index conversion: {e}"
        ) from e
