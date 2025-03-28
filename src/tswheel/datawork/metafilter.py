import pandas as pd
import collections.abc
from typing import Optional, Dict, Union  # Add Union, Iterable


def filter_df(
    df: pd.DataFrame,
    filters: Dict[str, Union[str, int, collections.abc.Iterable]],
    label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filters a DataFrame based on specified criteria and returns a summary DataFrame.

    Applies filters sequentially based on the provided `filters` dictionary.
    The resulting filtered DataFrame is stored within a new summary DataFrame.
    This summary DataFrame contains columns corresponding to the filter keys and
    their values, plus a 'result' column holding the filtered DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be filtered.
        filters (Dict[str, Union[str, int, collections.abc.Iterable]]):
            A dictionary defining the filtering criteria.
            Keys are column names (str) present in `df`.
            Values specify the criteria for filtering the corresponding column.
            Values can be:
            - A single string or integer to match exactly.
            - An iterable (like list, tuple, set) of strings or integers
              for matching any value within the iterable.
        label (Optional[str], optional): If provided, this string is used as the
            index label for the single row in the returned summary DataFrame.
            If None, pandas default integer index (0) is used. Defaults to None.

    Returns:
        pd.DataFrame: A summary DataFrame with one row. It includes columns for
            each filter criterion used and a 'result' column containing the
            final filtered pandas DataFrame. If the input `filters` dictionary
            is empty, the original `df` is returned with a warning.

    Raises:
        TypeError: If `df` is not a pandas DataFrame, `filters` is not a dictionary,
                   filter values are not of the expected types (str, int, iterable),
                   or `label` is provided but is not a string.
        ValueError: If a column specified in `filters` does not exist in `df`,
                    or if applying any filter step results in an empty DataFrame.

    Examples:
        >>> import pandas as pd
        >>> data = {'col_a': ['X', 'Y', 'X', 'Z', 'Y'],
        ...         'col_b': [1, 2, 3, 1, 2],
        ...         'value': [10, 20, 30, 40, 50]}
        >>> df = pd.DataFrame(data)
        >>> df
          col_a  col_b  value
        0     X      1     10
        1     Y      2     20
        2     X      3     30
        3     Z      1     40
        4     Y      2     50

        >>> # Basic filtering with a single value
        >>> filters1 = {'col_a': 'X'}
        >>> summary1 = filter_df(df, filters1, label='Filter_X')
        >>> print(summary1)
                   col_a                 result
        Filter_X     [X]  col_a  col_b  value
                         0     X      1     10
                         2     X      3     30
        >>> print(summary1.loc['Filter_X', 'result'])
          col_a  col_b  value
        0     X      1     10
        2     X      3     30

        >>> # Filtering with multiple values and a label
        >>> filters2 = {'col_a': ['X', 'Y'], 'col_b': 2}
        >>> summary2 = filter_df(df, filters2, label='Filter_XY_B2')
        >>> print(summary2)
                     col_a  col_b                result
        Filter_XY_B2 [X, Y]    [2] col_a  col_b  value
                                 1     Y      2     20
                                 4     Y      2     50
        >>> print(summary2.loc['Filter_XY_B2', 'result'])
          col_a  col_b  value
        1     Y      2     20
        4     Y      2     50

        >>> # Example raising ValueError due to empty result
        >>> try:
        ...     filter_df(df, {'col_a': 'NonExistent'})
        ... except ValueError as e:
        ...     print(e)
        Filtering on column 'col_a' with values ['NonExistent'] resulted in an empty DataFrame. No data matches the specified filter criteria.

        >>> # Example handling empty filters dictionary
        >>> empty_filters = {}
        >>> result_empty = filter_df(df, empty_filters)
        Warning: 'filters' dictionary is empty. Returning input 'df' unchanged.
        >>> print(result_empty)
          col_a  col_b  value
        0     X      1     10
        1     Y      2     20
        2     X      3     30
        3     Z      1     40
        4     Y      2     50
    """
    # --- Input Validation ---

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    # Validate filters type
    if not isinstance(filters, dict):
        raise TypeError("Input 'filters' must be a dictionary.")
    if not filters:
        print("Warning: 'filters' dictionary is empty. Returning input 'df' unchanged.")
        return df
    # Validate that all filter keys exist in the DataFrame and values are suitable for filtering
    for column, values in filters.items():
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        # Check if values is a list-like iterable (but not a string/int), a string, or an int
        is_iterable_non_scalar = isinstance(
            values, collections.abc.Iterable
        ) and not isinstance(values, str)
        is_scalar = isinstance(values, (str, int))  # Allow str and int directly

        if not (is_iterable_non_scalar or is_scalar):
            raise TypeError(
                f"Values for column '{column}' in 'filters' must be a list-like iterable "
                f"(e.g., list, tuple, set), a string, or an integer. Got type: {type(values).__name__}"
            )
    # Validate label type if provided
    if label is not None and not isinstance(label, str):
        raise TypeError("Input 'label' must be a string or None.")

    # --- Apply Filters ---

    # Start with the original DataFrame
    filtered_df = df.copy()

    # Apply each filter sequentially
    for column, values in filters.items():
        # If the value is a string or an int, wrap it in a list for isin()
        if isinstance(values, (str, int)):
            current_values = [values]
        else:
            # Otherwise, assume it's an iterable (list, tuple, set, etc.)
            current_values = values
        filtered_df = filtered_df[filtered_df[column].isin(current_values)]

        # Check if the DataFrame is empty after applying the filter
        if filtered_df.empty:
            raise ValueError(
                f"Filtering on column '{column}' with values {current_values} resulted in an empty DataFrame. "
                f"No data matches the specified filter criteria."
            )

    # --- Collect Results ---

    # Create a summary DataFrame with filter columns and the result
    summary_data = {column: [values] for column, values in filters.items()}
    summary_data["result"] = [filtered_df]
    summary_df = pd.DataFrame(summary_data)

    if label is not None:
        summary_df.index = [label]

    return summary_df


def metafilter_df(
    df: pd.DataFrame,
    labeled_filters: Dict[str, Dict[str, Union[str, int, collections.abc.Iterable]]],
) -> pd.DataFrame:
    """
    Applies multiple named sets of filters to a DataFrame using `filter_df`.

    Iterates through a dictionary of labeled filter sets (`labeled_filters`).
    For each label and its corresponding filter dictionary, it calls `filter_df`
    on the input DataFrame `df`. The resulting summary DataFrames (one for each
    filter set) are then concatenated vertically into a single DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame to apply filters to.
        labeled_filters (Dict[str, Dict[str, Union[str, int, collections.abc.Iterable]]]):
            A dictionary where:
            - Keys are string labels identifying each filter set.
            - Values are filter dictionaries, structured identically to the
              `filters` argument in the `filter_df` function.
            Example: {'Set1': {'col1': 'valA'}, 'Set2': {'col2': [1, 2], 'col3': 'valB'}}

    Returns:
        pd.DataFrame: A DataFrame resulting from the vertical concatenation of all
            summary DataFrames generated by applying each filter set via `filter_df`.
            Each row corresponds to one labeled filter set, inheriting the index label
            and containing the filter criteria columns and the 'result' column
            with the respective filtered DataFrame. If `labeled_filters` is empty,
            the original `df` is returned with a warning.

    Raises:
        TypeError: If `df` is not a pandas DataFrame, `labeled_filters` is not a
                   dictionary, keys in `labeled_filters` are not strings, or values
                   in `labeled_filters` are not dictionaries. Also propagates
                   TypeErrors raised by underlying `filter_df` calls.
        ValueError: Propagates ValueErrors raised by underlying `filter_df` calls
                    (e.g., column not found, filter results in empty DataFrame).

    Examples:
        >>> import pandas as pd
        >>> data = {'group': ['A', 'A', 'B', 'B', 'C'],
        ...         'type': ['X', 'Y', 'X', 'Y', 'X'],
        ...         'value': [10, 15, 20, 25, 30]}
        >>> df = pd.DataFrame(data)
        >>> df
          group type  value
        0     A    X     10
        1     A    Y     15
        2     B    X     20
        3     B    Y     25
        4     C    X     30

        >>> # Define multiple filter sets
        >>> filter_sets = {
        ...     'GroupA_TypeX': {'group': 'A', 'type': 'X'},
        ...     'GroupB': {'group': 'B'},
        ...     'TypeY': {'type': 'Y'}
        ... }

        >>> # Apply multiple filters
        >>> combined_summary = metafilter_df(df, filter_sets)
        >>> print(combined_summary)
                           group type                 result
        GroupA_TypeX         [A]  [X]  group type  value
                                    0     A    X     10
        GroupB               [B]  NaN  group type  value
                                    2     B    X     20
                                    3     B    Y     25
        TypeY                NaN  [Y]  group type  value
                                    1     A    Y     15
                                    3     B    Y     25

        >>> # Accessing a specific result DataFrame
        >>> print(combined_summary.loc['GroupB', 'result'])
          group type  value
        2     B    X     20
        3     B    Y     25

        >>> # Example with empty labeled_filters
        >>> empty_labeled = {}
        >>> result_empty = metafilter_df(df, empty_labeled)
        Warning: 'labeled_filters' dictionary is empty. Returning input 'df' unchanged.
        >>> print(result_empty)
          group type  value
        0     A    X     10
        1     A    Y     15
        2     B    X     20
        3     B    Y     25
        4     C    X     30
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(labeled_filters, dict):
        raise TypeError("Input 'labeled_filters' must be a dictionary.")
    if not labeled_filters:
        print(
            "Warning: 'labeled_filters' dictionary is empty. Returning input 'df' unchanged."
        )
        return df
    for label, filters_dict in labeled_filters.items():
        # Validate label and filters_dict types within the loop
        if not isinstance(label, str):
            raise TypeError(
                f"Key in 'labeled_filters' must be a string (label). Found: {label} ({type(label).__name__})"
            )
        if not isinstance(filters_dict, dict):
            raise TypeError(
                f"Value for label '{label}' in 'labeled_filters' must be a dictionary. Found: {type(filters_dict).__name__}"
            )
        # Deeper validation of filters_dict contents happens within filter_df

    # --- Apply Filters and Collect Results ---
    all_results = [
        filter_df(
            df=df,
            filters=filters_dict,
            label=label,
        )
        for label, filters_dict in labeled_filters.items()
    ]

    final_df = pd.concat(all_results)

    return final_df
