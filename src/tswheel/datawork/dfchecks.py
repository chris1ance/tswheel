import pandas as pd

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
