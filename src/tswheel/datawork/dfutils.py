import pandas as pd
from .dfchecks import have_same_index_type

pd.set_option("mode.copy_on_write", True)


def merge_on_index(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    suffixes: tuple[str, str] = ("_x", "_y"),
    validate: str | None = None,
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
    # Check if indices are compatible
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
