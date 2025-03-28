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
    dfs: Dict[str, pd.DataFrame],
    filters: Dict[str, Dict[str, Dict[str, Union[str, int, collections.abc.Iterable]]]],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Applies multiple sets of filters across multiple DataFrames based on cases.

    Iterates through cases defined in the `filters` dictionary. For each case,
    it applies specific filters (defined per DataFrame label within the case)
    to the corresponding DataFrames provided in the `dfs` dictionary, using the
    `filter_df` function. The results for each case are stored in a nested
    dictionary.

    Args:
        dfs (Dict[str, pd.DataFrame]): A dictionary where keys are string labels
            (e.g., "df1", "sales_data") and values are the corresponding
            pandas DataFrames to be filtered.
        filters (Dict[str, Dict[str, Dict[str, Union[str, int, collections.abc.Iterable]]]]):
            A nested dictionary defining the filtering operations for different cases.
            - Outer keys: String labels for each case (e.g., "HighValue_Customers", "Region_A").
            - Intermediate keys: String labels matching the keys in the `dfs` dictionary
              (e.g., "df1", "sales_data"), indicating which DataFrame the filters apply to.
            - Inner values: Filter dictionaries, structured identically to the
              `filters` argument in the `filter_df` function, specifying the
              actual filtering criteria for the respective DataFrame within that case.
            Example:
            {
                "case1": {
                    "df1": {"col_a": "X"},
                    "df2": {"col_b": [1, 2]},
                },
                "case2": {
                    "df1": {"col_c": "Y"},
                }
            }

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: A nested dictionary containing the results.
            - Outer keys: Case labels from the input `filters` dictionary.
            - Inner keys: DataFrame labels corresponding to the DataFrames filtered
              within that case.
            - Inner values: The summary DataFrames returned by `filter_df` for
              each specific DataFrame and filter set within the case. Each summary
              DataFrame has the case label as its index. If `filters` is empty,
              an empty dictionary is returned.

    Raises:
        TypeError: If `dfs` is not a dictionary, its values are not pandas DataFrames,
                   `filters` is not a dictionary, outer keys in `filters` are not
                   strings, intermediate keys are not strings, or intermediate
                   values are not dictionaries. Also propagates TypeErrors from
                   `filter_df`.
        ValueError: If an intermediate key (DataFrame label) in `filters` does not
                    exist as a key in `dfs`. Also propagates ValueErrors from
                    `filter_df` (e.g., column not found, filter results in
                    empty DataFrame).

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

        >>> # Define filters for different cases across the DataFrames
        >>> filter_config = {
        ...     "North_Region": {
        ...         "sales": {"Region": "North"},
        ...         "customers": {"Region": "North"}
        ...     },
        ...     "High_Sales_Segment_B": {
        ...         "sales": {"Sales": [150, 200]}, # South (150), North (200)
        ...         "customers": {"Segment": "B"} # North (2), North (4)
        ...     },
        ...     "West_Only_Sales": {
        ...         "sales": {"Region": "West"}
        ...     }
        ... }

        >>> # Apply the metafilters
        >>> results = metafilter_df(dfs_input, filter_config)

        >>> # Explore results for "North_Region" case
        >>> print("--- Case: North_Region ---")
        >>> print("Sales Data:")
        >>> print(results["North_Region"]["sales"])
                         Region                 result
        North_Region    [North]  Region  Sales  Units
                                0  North    100     10
                                2  North    200     20
        >>> print("\\nCustomer Data:")
        >>> print(results["North_Region"]["customers"])
                         Region                        result
        North_Region    [North]   CustomerID Region Segment
                                1           2  North       B
                                3           4  North       B

        >>> # Explore results for "High_Sales_Segment_B" case
        >>> print("\\n--- Case: High_Sales_Segment_B ---")
        >>> print("Sales Data (Sales >= 150):")
        >>> print(results["High_Sales_Segment_B"]["sales"])
                                   Sales                 result
        High_Sales_Segment_B  [150, 200]  Region  Sales  Units
                                         1  South    150     15
                                         2  North    200     20
        >>> print("\\nCustomer Data (Segment B):")
        >>> print(results["High_Sales_Segment_B"]["customers"])
                             Segment                        result
        High_Sales_Segment_B     [B]   CustomerID Region Segment
                                     1           2  North       B
                                     3           4  North       B

        >>> # Explore results for "West_Only_Sales" case (only sales filtered)
        >>> print("\\n--- Case: West_Only_Sales ---")
        >>> print("Sales Data:")
        >>> print(results["West_Only_Sales"]["sales"])
                             Region               result
        West_Only_Sales      [West]  Region  Sales  Units
                                    3   West     50      5
        >>> # "customers" key doesn't exist for this case in results
        >>> print("customers" in results["West_Only_Sales"])
        False

        >>> # Example with empty filters
        >>> empty_filters_config = {}
        >>> empty_results = metafilter_df(dfs_input, empty_filters_config)
        >>> print(empty_results)
        {}

        >>> # Example raising ValueError (df label mismatch)
        >>> invalid_filter_config = {"Bad_Case": {"non_existent_df": {"col": "val"}}}
        >>> try:
        ...     metafilter_df(dfs_input, invalid_filter_config)
        ... except ValueError as e:
        ...     print(e)
        DataFrame label 'non_existent_df' used in filters for case 'Bad_Case' does not exist in the input 'dfs' dictionary.

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

    # --- Apply Filters per Case and DataFrame ---
    outputs: Dict[str, Dict[str, pd.DataFrame]] = {}

    for case_label, case_filters in filters.items():
        case_outputs: Dict[str, pd.DataFrame] = {}

        for dflabel, dffilters in case_filters.items():
            # Apply the specific filters using filter_df, passing the case label
            # filter_df will raise errors if filters are invalid or result is empty
            try:
                df_extract = filter_df(
                    df=dfs[dflabel],
                    filters=dffilters,
                    label=case_label,  # Use the case label for the summary row index
                )
                case_outputs[dflabel] = df_extract
            except (ValueError, TypeError) as e:
                # Add context to errors raised by filter_df
                print(
                    f"Error applying filters for case '{case_label}', DataFrame '{dflabel}':"
                )
                raise e  # Re-raise the original error after printing context

        # Store the results for the current case
        outputs[case_label] = case_outputs

    return outputs
