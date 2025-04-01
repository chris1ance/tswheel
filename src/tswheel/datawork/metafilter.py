import pandas as pd
import collections.abc
from typing import Optional, Dict, Union, TypeAlias, Callable, Any
import copy

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

PostFilterFuncResult: TypeAlias = Dict[str, Dict[str, Any]]


class Metafilter:
    """
    Filter DataFrames using diverse criteria and optional post-processing.

    Provides static methods to filter a single DataFrame or apply multiple filter
    configurations (cases) across several DataFrames. An additional method allows
    applying custom post-processing functions to the filtered results.

    Public Methods:
        filter_df: Filter a single DataFrame based on specified criteria.
        metafilter_df: Apply multiple filter cases across multiple DataFrames.
        apply_post_filter_funcs: Apply post-filter functions to results from metafilter_df.
    """

    @staticmethod
    def filter_df(
        df: pd.DataFrame,
        filters: FilterDict,
        label: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filter a DataFrame based on specified criteria and return a summary DataFrame.

        Applies filters sequentially based on the provided `filters` dictionary.
        The resulting filtered DataFrame is stored within a new summary DataFrame.
        This summary DataFrame contains columns corresponding to the filter keys and
        their values, plus a 'result' column holding the filtered DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to be filtered.
            filters (FilterDict): A dictionary defining the filtering criteria.
                Keys are column names (str) present in `df`.
                Values (FilterValue) specify the criteria for filtering:
                - A single string or integer to match exactly.
                - An iterable (like list, tuple, set) of strings or integers
                  for matching any value within the iterable.
            label (Optional[str], default=None): If provided, this string is used as the
                index label for the single row in the returned summary DataFrame.
                If None, pandas default integer index (0) is used.

        Returns:
            summary_df (pd.DataFrame): A summary DataFrame with one row. It includes
                columns for each filter criterion used and a 'result' column containing
                the final filtered pandas DataFrame. If the input `filters` dictionary
                is empty, the original `df` is returned directly (with a warning).

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
            >>> summary1 = Metafilter.filter_df(df, filters1, label='Filter_X')
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
            >>> summary2 = Metafilter.filter_df(df, filters2, label='Filter_XY_B2')
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
            ...     Metafilter.filter_df(df, {'col_a': 'NonExistent'})
            ... except ValueError as e:
            ...     print(e)
            Filtering on column 'col_a' with values ['NonExistent'] resulted in an empty DataFrame. No data matches the specified filter criteria.

            >>> # Example handling empty filters dictionary
            >>> empty_filters: FilterDict = {}
            >>> result_empty = Metafilter.filter_df(df, empty_filters)
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
            print(
                "Warning: 'filters' dictionary is empty. Returning input 'df' unchanged."
            )
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

    @classmethod
    def metafilter_df(
        cls,
        dfs: DataFrameDict,
        filters: MetaFilterConfig,
    ) -> MetaFilterResult:
        """
        Apply multiple filter cases across multiple DataFrames.

        Iterates through filter cases defined in `filters`. For each case, applies the
        specified filters to the corresponding DataFrames in `dfs` using `filter_df`.
        This generates a summary DataFrame for each filtered DataFrame within each case,
        containing columns for the filter criteria and a 'result' column holding the
        actual filtered data. The output structure is suitable for optional
        post-processing with `apply_post_filter_funcs`.

        Args:
            dfs (DataFrameDict): A dictionary mapping string labels to pandas
                DataFrames to be filtered.
            filters (MetaFilterConfig): A nested dictionary defining filtering operations.
                Structure: {case_label: {df_label: {column_name: filter_value}}}

        Returns:
            outputs (MetaFilterResult): A nested dictionary containing the results.
                Structure: {case_label: {df_label: summary_dataframe}}
                - Outer keys: Case labels from the input `filters`.
                - Inner keys: DataFrame labels corresponding to DataFrames filtered
                  within that case.
                - Inner values: The summary DataFrame generated by `filter_df` for
                  that specific DataFrame and case.
                If `filters` is empty, an empty dictionary is returned.

        Raises:
            TypeError: If inputs `dfs` or `filters` have incorrect types or structures
                       (e.g., non-string keys, non-DataFrame/dict values).
            ValueError: If a `df_label` in `filters` does not exist as a key in `dfs`,
                        or propagated from `filter_df` if filtering results in an
                        empty DataFrame.

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
            >>> # Apply the metafiltering
            >>> results = Metafilter.metafilter_df(dfs_input, filter_config)

            >>> # Check results for "North_Region" case
            >>> print("--- Case: North_Region ---")
            >>> print(results["North_Region"]["sales"])
                           Region                 result
            North_Region  [North]  Region  Sales  Units
                                  0  North    100     10
                                  2  North    200     20
            >>> print(results["North_Region"]["customers"])
                          Region                    result
            North_Region  [North]   CustomerID Region Segment
                                  1           2  North       B
                                  3           4  North       B

            >>> # Check results for "High_Sales" case
            >>> print("\\n--- Case: High_Sales ---")
            >>> print(results["High_Sales"]["sales"])
                           Sales                 result
            High_Sales  [150, 200]  Region  Sales  Units
                                   1  South    150     15
                                   2  North    200     20
            >>> # This output 'results' can now be passed to apply_post_filter_funcs if needed.
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

        # --- Apply Filters per Case and per DataFrame ---
        outputs: MetaFilterResult = {}

        for case_label, case_filters in filters.items():
            case_outputs: Dict[str, pd.DataFrame] = {}

            for dflabel, dffilters in case_filters.items():
                try:
                    filter_df_result = cls.filter_df(
                        df=dfs[dflabel],
                        filters=dffilters,
                        label=case_label,  # Use the case label for the summary row index
                    )
                    case_outputs[dflabel] = filter_df_result
                except (ValueError, TypeError) as e:
                    # Add context to errors raised by filter_df or validation
                    print(
                        f"Error applying filters for case '{case_label}', DataFrame '{dflabel}':"
                    )
                    raise e  # Re-raise the original error after printing context

            # Store the results for the current case
            outputs[case_label] = case_outputs

        return outputs

    @classmethod
    def apply_post_filter_funcs(
        cls,
        metafilter_result: MetaFilterResult,
        post_filter_funcs: PostFilterFuncDict,
    ) -> PostFilterFuncResult:
        """
        Apply post-filter functions to the results obtained from `metafilter_df`.

        Iterates through the nested dictionary structure produced by `metafilter_df`.
        For each summary DataFrame within each case, it checks if a corresponding
        function exists in the `post_filter_funcs` dictionary based on the DataFrame's
        original label (`dflabel`). If a function is found, it is applied to a *copy*
        of that summary DataFrame. The output of the function replaces the original
        summary DataFrame in the results structure for that specific case and dflabel.

        Note: The functions provided in `post_filter_funcs` receive the entire one-row
        summary DataFrame (including filter columns and the 'result' column containing
        the filtered data) as input. They can return any value or structure, which will
        then be stored under the corresponding `dflabel` key within the case.

        Args:
            metafilter_result (MetaFilterResult): The nested dictionary output from
                `metafilter_df`. Structure: {case_label: {df_label: summary_dataframe}}
            post_filter_funcs (PostFilterFuncDict): A dictionary mapping DataFrame
                labels (str) to functions. Each function is applied to a copy of the
                summary DataFrame corresponding to its key within each case. The function
                must accept a pandas DataFrame (the summary DataFrame) as input.

        Returns:
            outputs (PostFilterFuncResult): A nested dictionary mirroring the structure
                of `metafilter_result`, but where the inner values associated with a
                `dflabel` that had a corresponding function in `post_filter_funcs`
                have been replaced by the output of that function. The type alias
                `PostFilterFuncResult` allows for arbitrary return types from the
                applied functions (`Any`).
                Structure: {case_label: {df_label: function_output}}

        Raises:
            TypeError: If `metafilter_result` or `post_filter_funcs` are not dictionaries,
                       if keys in `post_filter_funcs` are not strings, or if values
                       in `post_filter_funcs` are not callable.
            Exception: Propagated if a function provided in `post_filter_funcs` raises an
                       exception during its execution. The error message will include
                       context about the case and DataFrame label being processed.

        Examples:
            >>> # Setup similar to metafilter_df example
            >>> import pandas as pd
            >>> df_sales = pd.DataFrame({
            ...     'Region': ['North', 'South', 'North', 'West'],
            ...     'Sales': [100, 150, 200, 50], 'Units': [10, 15, 20, 5]
            ... })
            >>> df_customers = pd.DataFrame({
            ...     'CustomerID': [1, 2, 3, 4], 'Region': ['South', 'North', 'West', 'North'],
            ...     'Segment': ['A', 'B', 'A', 'B']
            ... })
            >>> dfs_input = {"sales": df_sales, "customers": df_customers}
            >>> filter_config: MetaFilterConfig = {
            ...     "North_Region": {"sales": {"Region": "North"}, "customers": {"Region": "North"}},
            ...     "High_Sales": {"sales": {"Sales": [150, 200]}}
            ... }
            >>> # First, get results from metafilter_df
            >>> initial_results = Metafilter.metafilter_df(dfs_input, filter_config)

            >>> # Define a post-filter function for the 'sales' DataFrame
            >>> # This function takes the summary_df and returns a count of filtered rows
            >>> def count_filtered_sales(summary_df: pd.DataFrame) -> str:
            ...     filtered_df = summary_df['result'].iloc[0]
            ...     return f"Count: {len(filtered_df)}"

            >>> # Define the dictionary linking the dflabel to the function
            >>> post_funcs: PostFilterFuncDict = {
            ...     "sales": count_filtered_sales
            ... }

            >>> # Apply the post-filter function
            >>> final_results = Metafilter.apply_post_filter_funcs(initial_results, post_funcs)

            >>> # Check the results - 'sales' entry is now the function output (a string)
            >>> print("--- Case: North_Region (Post-Processed) ---")
            >>> print(final_results["North_Region"]["sales"])
            Count: 2
            >>> # The 'customers' entry remains the original summary DataFrame
            >>> print(final_results["North_Region"]["customers"])
                          Region                    result
            North_Region  [North]   CustomerID Region Segment
                                  1           2  North       B
                                  3           4  North       B

            >>> print("\\n--- Case: High_Sales (Post-Processed) ---")
            >>> print(final_results["High_Sales"]["sales"])
            Count: 2
        """
        # --- Input Validation ---
        if not isinstance(metafilter_result, dict):
            raise TypeError("Input 'results' must be a dictionary.")

        if not isinstance(post_filter_funcs, dict):
            raise TypeError("Input 'post_filter_funcs' must be a dictionary.")

        for df_label, func in post_filter_funcs.items():
            if not isinstance(df_label, str):
                raise TypeError(
                    f"Keys in 'post_filter_funcs' must be strings (DataFrame labels). Found: {df_label} ({type(df_label).__name__})"
                )
            if not callable(func):
                raise TypeError(
                    f"Value for key '{df_label}' in 'post_filter_funcs' must be callable. Got type: {type(func).__name__}"
                )

        # --- Apply Post-Filter Functions ---
        # Create a copy of the results to avoid modifying the original
        outputs = copy.deepcopy(metafilter_result)

        # Iterate through the results
        for case_label, case_results in outputs.items():
            for dflabel, summary_df in case_results.items():
                if (
                    dflabel in post_filter_funcs
                ):  # Check if there's a specific func for this dflabel
                    func_to_apply = post_filter_funcs[dflabel]
                    try:
                        f_summary_df = func_to_apply(summary_df.copy())
                        outputs[case_label][dflabel] = f_summary_df
                    except Exception as func_error:
                        # Catch errors from the user's function and add context
                        print(
                            f"Error executing post-filter function for DataFrame label '{dflabel}' in case '{case_label}':"
                        )
                        raise func_error  # Re-raise the error from the function

        return outputs
