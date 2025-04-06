import pandas as pd
from typing import Literal
from pandas.api.types import is_numeric_dtype

pd.set_option("mode.copy_on_write", True)


def have_same_index_type(
    df1: pd.DataFrame | pd.Series,
    df2: pd.DataFrame | pd.Series,
) -> tuple[bool, str]:
    """
    Checks whether two pandas objects (DataFrame or Series) have the same type of index and optionally the same frequency.

    Args:
        df1 (pd.DataFrame | pd.Series): The first pandas object to check
        df2 (pd.DataFrame | pd.Series): The second pandas object to check

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
        >>> s1 = pd.Series([1,2,3], index=pd.date_range('2022-01-01', periods=3, freq='D'))
        >>> have_same_index_type(df1, s1)
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
    if not isinstance(df1, (pd.DataFrame, pd.Series)):
        raise TypeError("Input 'df1' must be a pandas DataFrame or Series.")
    if not isinstance(df2, (pd.DataFrame, pd.Series)):
        raise TypeError("Input 'df2' must be a pandas DataFrame or Series.")

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


def coerce_to_period_index(
    obj: pd.DataFrame | pd.Series,
    freq: Literal["B", "D", "W", "M", "Q", "Y"] | None = None,
) -> pd.DataFrame | pd.Series:
    """
    Ensures the pandas object's index is a pandas PeriodIndex with an allowed frequency.

    Converts the index if it is a DatetimeIndex, a compatible object/string
    index, or contains datetime.date objects. The resulting PeriodIndex must
    have a base frequency belonging to ['B', 'D', 'W', 'M', 'Q', 'Y'].

    - If the index is already a PeriodIndex with an allowed frequency, the
      object is returned unchanged.
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
        obj (pd.DataFrame | pd.Series): The pandas object to check and potentially convert.
        freq (str | None, optional): The target frequency for the PeriodIndex.
            Allowed values: 'B' (Business Day), 'D' (Calendar Day), 'W' (Weekly),
            'M' (Month), 'Q' (Quarter), 'Y' (Year).
            If None (default), frequency is inferred during conversion.
            If provided, this frequency is used.

    Returns:
        pd.DataFrame | pd.Series: A copy of the object with a PeriodIndex, or the
                                   original object if it already had a PeriodIndex
                                   with an allowed frequency. The returned type
                                   matches the input type.

    Raises:
        ValueError: If frequency inference fails (e.g., irregular DatetimeIndex)
                    and `freq` is not provided, or if conversion to PeriodIndex
                    fails after intermediate DatetimeIndex conversion.
        TypeError: If the input is not a pandas DataFrame or Series, if the index is a
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
        >>> df_period_dt = coerce_to_period_index(df_dt.copy())
        >>> print(isinstance(df_period_dt.index, pd.PeriodIndex))
        True
        >>> print(df_period_dt.index.freqstr)
        D
        >>>
        >>> # Series with DatetimeIndex
        >>> s_dt = pd.Series([1, 2, 3], index=dates)
        >>> s_period_dt = coerce_to_period_index(s_dt.copy())
        >>> print(isinstance(s_period_dt.index, pd.PeriodIndex))
        True
        >>> print(s_period_dt.index.freqstr)
        D
        >>>
        >>> # Object with PeriodIndex (no change)
        >>> periods = pd.period_range('2023-01', periods=3, freq='ME') # Use 'ME'
        >>> df_p = pd.DataFrame({'data': [4, 5, 6]}, index=periods)
        >>> df_p_checked = coerce_to_period_index(df_p)
        >>> print(df_p_checked.index is df_p.index)
        True
        >>> s_p = pd.Series([4, 5, 6], index=periods)
        >>> s_p_checked = coerce_to_period_index(s_p)
        >>> print(s_p_checked.index is s_p.index)
        True
        >>>
        >>> # Object with string index (convertible)
        >>> str_idx = ['2023-01', '2023-02', '2023-03']
        >>> df_str = pd.DataFrame({'data': [7, 8, 9]}, index=str_idx)
        >>> df_period_str = coerce_to_period_index(df_str.copy()) # Infer freq='ME'
        >>> print(isinstance(df_period_str.index, pd.PeriodIndex))
        True
        >>> print(df_period_str.index.freqstr)
        ME
        >>>
        >>> # Object with datetime.date index
        >>> date_idx = [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
        >>> df_date = pd.DataFrame({'data': [10, 11, 12]}, index=date_idx)
        >>> df_period_date = coerce_to_period_index(df_date.copy(), freq='ME') # Specify freq 'ME'
        >>> print(isinstance(df_period_date.index, pd.PeriodIndex))
        True
        >>> print(df_period_date.index.freqstr)
        ME
        >>>
        >>> # Object with non-convertible numeric index (raises TypeError)
        >>> df_range = pd.DataFrame({'data': [13, 14, 15]}, index=pd.RangeIndex(3))
        >>> try:
        ...     coerce_to_period_index(df_range)
        ... except TypeError as e:
        ...     print(e) # doctest: +ELLIPSIS
        Index type RangeIndex with dtype int64 cannot be converted to PeriodIndex...
        >>>
        >>> # Object with DatetimeIndex inferring unsupported freq (raises AssertionError)
        >>> irregular_dates = pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-05'])
        >>> df_irreg = pd.DataFrame({'data': [1, 2, 3]}, index=irregular_dates)
        >>> try:
        ...     coerce_to_period_index(df_irreg.copy()) # Infers '2D', which is not allowed
        ... except AssertionError as e:
        ...     print(e) # doctest: +ELLIPSIS
        Resulting PeriodIndex frequency '2D' (base: '2D') is not in ['B', 'D', 'W', 'M', 'Q', 'Y']
        >>>
        >>> # Force allowed frequency for irregular dates
        >>> df_irreg_period_forced_d = coerce_to_period_index(df_irreg.copy(), freq='D') # Force 'D'
        >>> print(df_irreg_period_forced_d.index)
        PeriodIndex(['2023-01-01', '2023-01-03', '2023-01-05'], dtype='period[D]')
    """
    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        raise TypeError("Input must be a pandas DataFrame or Series.")

    # Define the allowed base frequencies using preferred aliases
    ALLOWED_BASE_FREQS = ["B", "D", "W", "M", "Q", "Y"]

    # 1. Check for definitely non-convertible numeric index types
    if is_numeric_dtype(obj.index):
        raise TypeError(
            f"Index type {type(obj.index).__name__} with dtype {obj.index.dtype} "
            "cannot be converted to PeriodIndex. Only DatetimeIndex or potentially convertible "
            "object/string/date indexes are supported."
        )

    # 2. Check if already PeriodIndex
    if isinstance(obj.index, pd.PeriodIndex):
        # Ensure the existing frequency is valid according to the new rule
        assert obj.index.freq in ALLOWED_BASE_FREQS, (
            f"Existing PeriodIndex frequency '{obj.index.freqstr}' is not in {ALLOWED_BASE_FREQS}"
        )
        # Return the original object as no copy is needed
        return obj

    # 3. Check if DatetimeIndex
    if isinstance(obj.index, pd.DatetimeIndex):
        try:
            # Use provided freq if available, otherwise attempt inference
            new_index = obj.index.to_period(freq=freq)
            # Create a copy to avoid modifying the original object
            obj_copy = obj.copy()
            obj_copy.index = new_index
            # Assert the resulting frequency is valid
            assert obj_copy.index.freq in ALLOWED_BASE_FREQS, (
                f"Resulting PeriodIndex frequency '{obj_copy.index.freqstr}' is not in {ALLOWED_BASE_FREQS}"
            )
            return obj_copy
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
        datetime_index = pd.to_datetime(obj.index)
        # Now convert the intermediate DatetimeIndex to PeriodIndex
        try:
            new_index = datetime_index.to_period(freq=freq)
            # Create a copy
            obj_copy = obj.copy()
            obj_copy.index = new_index
            # Assert the resulting frequency is valid
            assert obj_copy.index.freq in ALLOWED_BASE_FREQS, (
                f"Resulting PeriodIndex frequency '{obj_copy.index.freqstr}' is not in {ALLOWED_BASE_FREQS}"
            )
            return obj_copy
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
            f"Index type {type(obj.index).__name__} cannot be converted to PeriodIndex. "
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


def merge_on_index(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    suffixes: tuple[str, str] = ("_left", "_right"),
    validate: Literal["1:1", "1:m", "m:1", "m:m"] | None = None,
    indicator: bool = False,
) -> pd.DataFrame:
    """
    Merges two DataFrames on their indices after validating that the indices are compatible.

    Args:
        left (pd.DataFrame): Left DataFrame to merge
        right (pd.DataFrame): Right DataFrame to merge
        how (str): Type of merge to perform ('inner', 'outer', 'left', 'right', 'cross')
        suffixes (tuple[str, str]): Suffixes to apply to overlapping column names
        validate (str): If specified, checks if merge is of specified type
            ('1:1', '1:m', 'm:1', 'm:m')
        indicator (bool): If True, adds a column indicating the source of each row

    Returns:
        pd.DataFrame: Merged DataFrame

    Raises:
        ValueError: If the indices are not compatible

    Examples:
        >>> import pandas as pd
        >>> df1 = pd.DataFrame({'a': [1, 2, 3]},
        ...                    index=pd.date_range('2022-01-01', periods=3, freq='D'))
        >>> df2 = pd.DataFrame({'b': [4, 5, 6]},
        ...                    index=pd.date_range('2022-01-02', periods=3, freq='D'))
        >>> merge_on_index(df1, df2)
           a  b
        2022-01-02  2  4
        2022-01-03  3  5

        >>> # Different index types will raise an error
        >>> df3 = pd.DataFrame({'c': [7, 8, 9]}, index=range(3))
        >>> merge_on_index(df1, df3)
        Traceback (most recent call last):
        ...
        ValueError: Index types differ: DatetimeIndex vs RangeIndex
    """

    # Check if validation is required - perform basic validation
    if validate:
        # For 1:1 validation, check for duplicates in indices
        if validate == "1:1":
            if left.index.duplicated().any():
                raise ValueError(
                    "Left index contains duplicate entries, validation '1:1' failed"
                )
            if right.index.duplicated().any():
                raise ValueError(
                    "Right index contains duplicate entries, validation '1:1' failed"
                )

    # First, validate that the indices are of the same type
    compatible, message = have_same_index_type(left, right)
    if not compatible:
        raise ValueError(message)

    # Perform the merge on indices
    return pd.merge(
        left=left,
        right=right,
        how=how,
        left_index=True,
        right_index=True,
        suffixes=suffixes,
        validate=validate,
        indicator=indicator,
    )


def ts_concat(
    objs: pd.DataFrame | pd.Series | list[pd.DataFrame | pd.Series],
    axis: Literal[0, 1, "index", "columns"] = 0,
    join: Literal["inner", "outer"] = "outer",
    ignore_index: bool = False,
) -> pd.DataFrame | pd.Series:
    """
    Concatenates a sequence of pandas objects along an axis after validation.

    If a single DataFrame or Series is passed (either directly or in a list of length 1),
    it is returned directly without concatenation or validation beyond type checking.

    For lists with multiple objects:
    Validates that all objects have the same index type and that no object
    has duplicate index values. If concatenating along columns (axis=1),
    it also validates that there are no overlapping column names between
    the DataFrames.

    Args:
        objs (pd.DataFrame | pd.Series | list[pd.DataFrame | pd.Series]):
            A single pandas Series or DataFrame, or a sequence of them to concatenate.
        axis (Literal[0, 1, 'index', 'columns'], default 0): The axis to
            concatenate along. 0/'index' for rows, 1/'columns' for columns.
            Ignored if only a single object is provided.
        join (Literal['inner', 'outer'], default 'outer'): How to handle
            indexes on the other axis (or axes). 'outer' for union, 'inner'
            for intersection. Ignored if only a single object is provided.
        ignore_index (bool, default False): If True, do not use the index
            values along the concatenation axis. The resulting axis will be
            labeled 0, ..., n - 1. Ignored if only a single object is provided.

    Returns:
        pd.DataFrame | pd.Series: The concatenated pandas object, or the single
            input object if only one was provided. Type depends on the input
            objects and the axis of concatenation.

    Raises:
        TypeError: If objs is not a pandas object or a list of pandas objects,
                   or if a list contains non-pandas objects.
        ValueError: If objs is an empty list, if any object in a list of multiple
                    objects has duplicate index values, if objects in a list
                    have incompatible index types, or if concatenating
                    multiple objects along columns and duplicate column names exist.

    Examples:
        >>> import pandas as pd
        >>>
        >>> # Example with single DataFrame input
        >>> df_single = pd.DataFrame({'A': [1, 2]})
        >>> result_single_df = ts_concat(df_single)
        >>> print(result_single_df is df_single)
        True
        >>>
        >>> # Example with list containing single Series
        >>> s_single_in_list = pd.Series(['a', 'b'])
        >>> result_single_s = ts_concat([s_single_in_list])
        >>> print(result_single_s is s_single_in_list)
        True
        >>>
        >>> # Concatenate multiple Series (rows, axis=0)
        >>> s1 = pd.Series(['a', 'b'], index=pd.RangeIndex(2))
        >>> s2 = pd.Series(['c', 'd'], index=pd.RangeIndex(start=2, stop=4))
        >>> ts_concat([s1, s2])
        0    a
        1    b
        2    c
        3    d
        dtype: object
        >>>
        >>> # Concatenate multiple DataFrames (columns, axis=1)
        >>> df1 = pd.DataFrame({'A': [1, 2]}, index=pd.RangeIndex(2))
        >>> df2 = pd.DataFrame({'B': [3, 4]}, index=pd.RangeIndex(2))
        >>> ts_concat([df1, df2], axis=1)
           A  B
        0  1  3
        1  2  4
        >>>
        >>> # Example with incompatible index types (raises ValueError)
        >>> df3 = pd.DataFrame({'C': [5, 6]}, index=pd.Index(['x', 'y']))
        >>> try:
        ...     ts_concat([df1, df3])
        ... except ValueError as e:
        ...     print(e)
        Index types differ: RangeIndex vs Index
        >>>
        >>> # Example with duplicate index (raises ValueError)
        >>> df_dup_idx = pd.DataFrame({'A': [7, 8]}, index=pd.RangeIndex(start=0, stop=2))
        >>> try:
        ...     ts_concat([df1, df_dup_idx]) # df1 and df_dup_idx share index 0, 1
        ... except ValueError as e:
        ...     print(e) # Note: pandas.concat doesn't raise by default, but our check does
        Object at index 1 has duplicate index values.
        >>>
        >>> # Example with duplicate columns when axis=1 (raises ValueError)
        >>> df_dup_col = pd.DataFrame({'A': [9, 10]}, index=pd.RangeIndex(2))
        >>> try:
        ...     ts_concat([df1, df_dup_col], axis=1)
        ... except ValueError as e:
        ...     print(e)
        Duplicate column names found: {'A'}

    See Also:
        pandas.concat: The underlying concatenation function.
    """
    # --- Input Validation ---

    # Handle single object input directly
    if isinstance(objs, (pd.DataFrame, pd.Series)):
        return objs

    # Validate list input
    if not isinstance(objs, list):
        raise TypeError(
            "Input 'objs' must be a pandas DataFrame, a pandas Series, or a list of them."
        )
    if not objs:
        raise ValueError("Input 'objs' list cannot be empty.")

    # Handle list with a single object
    if len(objs) == 1:
        single_obj = objs[0]
        if not isinstance(single_obj, (pd.DataFrame, pd.Series)):
            raise TypeError(
                "The single item in the 'objs' list must be a pandas DataFrame or Series."
            )
        return single_obj

    # --- Validation for List with Multiple Objects ---
    # Check for duplicate indices, compatible index types, and column name collisions
    first_obj = objs[0]
    if not isinstance(first_obj, (pd.DataFrame, pd.Series)):
        raise TypeError("Object at index 0 is not a pandas DataFrame or Series.")
    if first_obj.index.duplicated().any():
        raise ValueError("Object at index 0 has duplicate index values.")

    all_columns = set()
    if isinstance(first_obj, pd.DataFrame) and axis in [1, "columns"]:
        all_columns.update(first_obj.columns)
    elif isinstance(first_obj, pd.Series) and axis in [1, "columns"]:
        if first_obj.name is None:
            raise ValueError(
                "Series at index 0 must have a 'name' attribute for column concatenation."
            )
        all_columns.add(first_obj.name)

    for i, obj in enumerate(objs[1:], start=1):  # Start from the second object
        if not isinstance(obj, (pd.DataFrame, pd.Series)):
            raise TypeError(f"Object at index {i} is not a pandas DataFrame or Series.")
        if obj.index.duplicated().any():
            raise ValueError(f"Object at index {i} has duplicate index values.")

        # Check index compatibility against the first object
        compatible, message = have_same_index_type(first_obj, obj)
        if not compatible:
            raise ValueError(message)

        # Checks specific to column concatenation (axis=1)
        if axis in [1, "columns"]:
            if isinstance(obj, pd.DataFrame):
                current_columns = set(obj.columns)
                intersection = all_columns.intersection(current_columns)
                if intersection:
                    raise ValueError(f"Duplicate column names found: {intersection}")
                all_columns.update(current_columns)
            elif isinstance(obj, pd.Series):
                if obj.name is None:
                    raise ValueError(
                        f"Series at index {i} must have a 'name' attribute for column concatenation."
                    )
                if obj.name in all_columns:
                    raise ValueError(
                        f"Duplicate column name found: '{obj.name}' (from Series at index {i})"
                    )
                all_columns.add(obj.name)

    # --- Concatenation ---
    return pd.concat(
        objs=objs,
        axis=axis,
        join=join,
        ignore_index=ignore_index,
        sort=True if axis in [1, "columns"] else False,  # Sort index when axis=1
    )
