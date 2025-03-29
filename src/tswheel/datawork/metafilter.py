import pandas as pd
import collections.abc
from typing import Optional, Dict, Union, TypeAlias, Callable


# --- Type Aliases for Clarity ---

FilterValue: TypeAlias = Union[str, int, collections.abc.Iterable]
"""Type alias for acceptable filter values: a single string/int or an iterable."""

FilterDict: TypeAlias = Dict[str, FilterValue]
"""Type alias for a dictionary mapping column names to filter values."""

DataFrameDict: TypeAlias = Dict[str, pd.DataFrame]
"""Type alias for a dictionary mapping string labels to DataFrames."""

CaseFilterDict: TypeAlias = Dict[str, FilterDict]
"""Type alias for a dictionary mapping DataFrame labels to their specific filter dicts within a case."""

MetaFilterConfig: TypeAlias = Dict[str, CaseFilterDict]
"""Type alias for the nested dictionary structure defining metafiltering cases and their filters."""

PostFilterFuncDict: TypeAlias = Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]
"""Type alias for a dictionary mapping DataFrame labels (str) to functions 
that process a DataFrame and return a DataFrame."""

MetaFilterResult: TypeAlias = Dict[str, Dict[str, pd.DataFrame]]
"""Type alias for the nested dictionary structure returned by metafilter_df."""


def filter_df(
    df: pd.DataFrame,
    filters: FilterDict,
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
        filters (FilterDict):
            A dictionary defining the filtering criteria.
            Keys are column names (str) present in `df`.
            Values (FilterValue) specify the criteria for filtering:
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
                   filter values do not conform to FilterValue, or `label` is
                   provided but is not a string.
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
        >>> filters1: FilterDict = {'col_a': 'X'}
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
        >>> filters2: FilterDict = {'col_a': ['X', 'Y'], 'col_b': 2}
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
        >>> empty_filters: FilterDict = {}
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
        return df.copy()
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
    dfs: DataFrameDict,
    filters: MetaFilterConfig,
    post_filter_funcs: Optional[PostFilterFuncDict] = None,
) -> MetaFilterResult:
    """
    Apply multiple filters across DataFrames based on cases, optionally applying functions.

    Iterates through filter cases defined in `filters`. For each case, applies the
    specified filters to the corresponding DataFrames in `dfs` using `filter_df`.
    This generates a summary DataFrame for each filtered DataFrame, containing columns
    for the filter criteria and a 'result' column holding the actual filtered data.

    If `post_filter_funcs` is provided, it then iterates through the collected summary
    DataFrames. For each summary DataFrame, if a function exists in `post_filter_funcs`
    matching the DataFrame's original label (`dflabel`), that function is applied
    to a copy of the *entire* summary DataFrame (including filter columns and the
    'result' column). The function must take a pandas DataFrame (the summary DataFrame)
    as input and return a pandas DataFrame. The returned DataFrame is then stored in a
    *new column* named 'result_post_filter_func' within the original summary DataFrame structure.

    Args:
        dfs (DataFrameDict):
            A dictionary mapping string labels to pandas DataFrames to be filtered.
        filters (MetaFilterConfig):
            A nested dictionary defining filtering operations.
            Structure: {case_label: {df_label: {column_name: filter_value}}}
        post_filter_funcs (Optional[PostFilterFuncDict], default=None):
            A dictionary mapping DataFrame labels (str) to functions. Each function
            is applied to a copy of the *summary DataFrame* corresponding to its key
            *after* the initial filtering. The function must take a pandas DataFrame
            as input and return a pandas DataFrame. Its output is stored in the
            'result_post_filter_func' column. If None, or if a df_label has no
            corresponding function, this column is not added for that summary DataFrame.

    Returns:
        MetaFilterResult:
            A nested dictionary containing the results.
            Structure: {case_label: {df_label: summary_dataframe}}
            - Outer keys: Case labels from the input `filters`.
            - Inner keys: DataFrame labels corresponding to DataFrames filtered
              within that case.
            - Inner values: The summary DataFrame generated by `filter_df`. If a
              post-filter function was applied, this DataFrame will contain an
              additional 'result_post_filter_func' column holding the output
              of that function.
            If `filters` is empty, an empty dictionary is returned.

    Raises:
        TypeError: If inputs have incorrect types/structures, or if a function
                   in `post_filter_funcs` is not callable or returns the wrong type.
        ValueError: If labels in `filters` don't exist in `dfs`, or propagated
                    from `filter_df`.
        KeyError: If accessing data using `.loc` fails during post-processing.
        Exception: Propagated if a function provided in `post_filter_funcs` raises an
                   exception during its execution.

    Examples:
        >>> import pandas as pd
        >>> df_sales = pd.DataFrame({
        ...     'Region': ['North', 'South', 'North', 'West'],
        ...     'Sales': [100, 150, 200, 50],
        ...     'Units': [10, 15, 20, 5]
        ... })
        >>> df_customers = pd.DataFrame({
        ...     'CustomerID': [1, 2, 3, 4],
        ...     'Region': ['South', 'North', 'West', 'North'],
        ...     'Segment': ['A', 'B', 'A', 'B']
        ... })
        >>> dfs_input = {"sales": df_sales, "customers": df_customers}
        >>> filter_config: MetaFilterConfig = {
        ...     "North_Region": {"sales": {"Region": "North"}, "customers": {"Region": "North"}},
        ...     "High_Sales": {"sales": {"Sales": [150, 200]}}
        ... }

        >>> # Post-filter function processes the summary_df and returns a new DataFrame
        >>> def process_summary_and_return_status(summary_df: pd.DataFrame) -> pd.DataFrame:
        ...     # summary_df is the output of filter_df
        ...     # Example: Create a status DataFrame based on the input summary
        ...     case_lbl = summary_df.index[0]
        ...     status_msg = f"Summary processed for case: {case_lbl}"
        ...     if 'result' in summary_df.columns:
        ...         status_msg += f", Filtered records: {len(summary_df['result'].iloc[0])}"
        ...     return pd.DataFrame({'status': [status_msg]}) # Function must return a DataFrame
        >>>
        >>> post_funcs: PostFilterFuncDict = {
        ...     "sales": process_summary_and_return_status
        ... }

        >>> results_post = metafilter_df(dfs_input, filter_config, post_filter_funcs=post_funcs)

        >>> # Check results for "North_Region" case - sales summary has new column
        >>> print("--- Case: North_Region ---")
        >>> north_sales_summary = results_post["North_Region"]["sales"]
        >>> print(north_sales_summary)
                         Region                 result result_post_filter_func
        North_Region    [North]  Region  Sales  Units                  status
                                0  North    100     10  Summary processed for case: North_Region, Filt...
                                2  North    200     20
        >>> # Access the DataFrame stored in the new column
        >>> print(north_sales_summary['result_post_filter_func'].iloc[0])
                                                 status
        0  Summary processed for case: North_Region, Filtered records: 2
        >>> # Customer summary was not transformed, lacks the new column
        >>> print(results_post["North_Region"]["customers"])
                        Region                    result
        North_Region    [North]   CustomerID Region Segment
                                1           2  North       B
                                3           4  North       B

        >>> # Check results for "High_Sales" case - sales summary has new column
        >>> print("\\n--- Case: High_Sales ---")
        >>> print(results_post["High_Sales"]["sales"]['result_post_filter_func'].iloc[0])
                                                status
        0  Summary processed for case: High_Sales, Filtered records: 2
    """
    # --- Input Validation ---
    if not isinstance(dfs, dict):
        raise TypeError("Input 'dfs' must be a dictionary.")
    for df_label, df_obj in dfs.items():
        if not isinstance(df_label, str):
            raise TypeError(
                f"Keys in 'dfs' must be strings. Found: {df_label} ({type(df_label).__name__})"
            )
        if not isinstance(df_obj, pd.DataFrame):
            raise TypeError(
                f"Value for key '{df_label}' in 'dfs' must be a pandas DataFrame. Found: {type(df_obj).__name__}"
            )
    if not isinstance(filters, dict):
        raise TypeError("Input 'filters' must be a dictionary.")

    # Validate nested structure of filters
    for case_label, case_filters_dict in filters.items():
        if not isinstance(case_label, str):
            raise TypeError(
                f"Outer keys (case labels) in 'filters' must be strings. Found: {case_label} ({type(case_label).__name__})"
            )
        if not isinstance(case_filters_dict, dict):
            raise TypeError(
                f"Value for case '{case_label}' in 'filters' must be a dictionary. Found: {type(case_filters_dict).__name__}"
            )

        for df_label, df_specific_filters in case_filters_dict.items():
            if not isinstance(df_label, str):
                raise TypeError(
                    f"Intermediate keys (DataFrame labels) in filters for case '{case_label}' must be strings. Found: {df_label} ({type(df_label).__name__})"
                )
            # Check if the DataFrame label from filters exists in the input dfs dictionary
            if df_label not in dfs:
                raise ValueError(
                    f"DataFrame label '{df_label}' used in filters for case '{case_label}' does not exist in the input 'dfs' dictionary."
                )
            if not isinstance(df_specific_filters, dict):
                raise TypeError(
                    f"Inner value (filter dict) for DataFrame '{df_label}' in case '{case_label}' must be a dictionary. Found: {type(df_specific_filters).__name__}"
                )

    # Validate post_filter_funcs if provided
    if post_filter_funcs is not None:
        if not isinstance(post_filter_funcs, dict):
            raise TypeError("Input 'post_filter_funcs' must be a dictionary or None.")
        for df_label, func in post_filter_funcs.items():
            if not isinstance(df_label, str):
                raise TypeError(
                    f"Keys in 'post_filter_funcs' must be strings (DataFrame labels). Found: {df_label} ({type(df_label).__name__})"
                )
            if not callable(func):
                raise TypeError(
                    f"Value for key '{df_label}' in 'post_filter_funcs' must be callable. Got type: {type(func).__name__}"
                )

    # --- Apply Initial Filters per Case and DataFrame ---
    outputs: MetaFilterResult = {}

    for case_label, case_filters in filters.items():
        case_outputs: Dict[str, pd.DataFrame] = {}

        for dflabel, dffilters in case_filters.items():
            # Apply the specific filters using filter_df
            try:
                df_extract = filter_df(
                    df=dfs[dflabel],
                    filters=dffilters,
                    label=case_label,  # Use the case label for the summary row index
                )
                case_outputs[dflabel] = df_extract
            except (ValueError, TypeError) as e:
                # Add context to errors raised by filter_df or validation
                print(
                    f"Error applying filters for case '{case_label}', DataFrame '{dflabel}':"
                )
                raise e  # Re-raise the original error after printing context

        # Store the results for the current case
        outputs[case_label] = case_outputs

    # --- Optionally Apply Post-Filter Functions ---
    if post_filter_funcs:
        # Iterate through the results we just collected
        for case_label, case_results in outputs.items():
            for dflabel, summary_df in case_results.items():
                # Check if there's a specific function for this dflabel
                if dflabel in post_filter_funcs:
                    func_to_apply = post_filter_funcs[dflabel]
                    try:
                        # --- Apply func to the whole summary DataFrame ---
                        transformed_df = func_to_apply(summary_df.copy())

                        # Validate the output of the post_filter_func
                        if not isinstance(transformed_df, pd.DataFrame):
                            if isinstance(transformed_df, pd.Series):
                                transformed_df = transformed_df.to_frame()
                            else:
                                raise TypeError(
                                    f"The function for '{dflabel}' in 'post_filter_funcs' must return a pandas DataFrame or Series. "
                                    f"For case '{case_label}', it returned type: {type(transformed_df).__name__}"
                                )

                        # Store result in a new column
                        outputs[case_label][dflabel]["result_post_filter_func"] = [
                            transformed_df
                        ]

                    except Exception as func_error:
                        # Catch errors from the user's function and add context
                        print(
                            f"Error executing post-filter function for DataFrame label '{dflabel}' in case '{case_label}':"
                        )
                        raise func_error  # Re-raise the error from the function

    return outputs
