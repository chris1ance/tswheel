import pandas as pd
import collections.abc
from typing import Dict, Union, TypeAlias, Callable, Any
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

    Provides static methods to apply multiple filter configurations (cases) across
    several DataFrames. An additional method allows applying custom post-processing
    functions to the filtered results.

    Public Methods:
        metafilter_df: Apply multiple filter cases across multiple DataFrames.
        apply_post_filter_funcs: Apply post-filter functions to results from metafilter_df.
    """

    @staticmethod
    def metafilter_df(
        dfs: DataFrameDict,
        filters: MetaFilterConfig,
    ) -> MetaFilterResult:
        """
        Apply multiple filter cases across multiple DataFrames.

        Iterates through filter cases defined in `filters`. For each case, applies the
        specified filters sequentially to the corresponding DataFrames in `dfs`.
        The output is a nested dictionary containing the directly filtered DataFrames.
        This structure is suitable for optional post-processing with
        `apply_post_filter_funcs`.

        Args:
            dfs (DataFrameDict): A dictionary mapping string labels to pandas
                DataFrames to be filtered.
            filters (MetaFilterConfig): A nested dictionary defining filtering operations.
                Structure: {case_label: {df_label: {column_name: filter_value}}}

        Returns:
            outputs (MetaFilterResult): A nested dictionary containing the filtered results.
                Structure: {case_label: {df_label: filtered_dataframe}}
                - Outer keys: Case labels from the input `filters`.
                - Inner keys: DataFrame labels corresponding to DataFrames filtered
                  within that case.
                - Inner values: The resulting filtered pandas DataFrame for that
                  specific DataFrame and case.
                If `filters` is empty, an empty dictionary is returned.

        Raises:
            TypeError: If inputs `dfs` or `filters` have incorrect types or structures
                       (e.g., non-string keys, non-DataFrame/dict values).
            ValueError: If a `df_label` in `filters` does not exist as a key in `dfs`.

        Examples:
            >>> import pandas as pd
            >>> from typing import Dict, Union, TypeAlias, Callable, Any # For type hints in example
            >>> FilterValue: TypeAlias = Union[str, int, collections.abc.Iterable]
            >>> FilterDict: TypeAlias = Dict[str, FilterValue]
            >>> DataFrameDict: TypeAlias = Dict[str, pd.DataFrame]
            >>> CaseFilterDict: TypeAlias = Dict[str, FilterDict]
            >>> MetaFilterConfig: TypeAlias = Dict[str, CaseFilterDict]
            >>> MetaFilterResult: TypeAlias = Dict[str, Dict[str, pd.DataFrame]]
            >>> PostFilterFuncDict: TypeAlias = Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]
            >>> PostFilterFuncResult: TypeAlias = Dict[str, Dict[str, Any]]
            >>>
            >>> # Example DataFrames
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
            >>> dfs_input: DataFrameDict = {"sales": df_sales, "customers": df_customers}
            >>>
            >>> # Example Filter Configuration
            >>> filter_config: MetaFilterConfig = {
            ...     "North_Region": {"sales": {"Region": "North"}, "customers": {"Region": "North"}},
            ...     "High_Sales_West_Customers": {"sales": {"Sales": [150, 200]}, "customers": {"Region": "West"}}
            ... }
            >>>
            >>> # Apply the metafiltering
            >>> results = Metafilter.metafilter_df(dfs_input, filter_config)
            >>>
            >>> # Check results for "North_Region" case
            >>> print("--- Case: North_Region ---")
            >>> print("Sales (filtered):")
            >>> print(results["North_Region"]["sales"])
            >>> print("\nCustomers (filtered):")
            >>> print(results["North_Region"]["customers"])
            >>>
            >>> # Check results for "High_Sales_West_Customers" case
            >>> print("\n--- Case: High_Sales_West_Customers ---")
            >>> print("Sales (filtered):")
            >>> print(results["High_Sales_West_Customers"]["sales"])
            >>> print("\nCustomers (filtered):")
            >>> print(results["High_Sales_West_Customers"]["customers"])
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
        outputs: MetaFilterResult = copy.deepcopy(filters)

        for case_label, case_filters in filters.items():
            case_outputs: Dict[str, pd.DataFrame] = {}

            for df_label, df_filters in case_filters.items():
                filtered_df = dfs[df_label].copy()

                # Apply each filter sequentially
                for column, column_values in df_filters.items():
                    column_values_list = (
                        [column_values]
                        if isinstance(column_values, (str, int))
                        else column_values
                    )
                    filtered_df = filtered_df[
                        filtered_df[column].isin(column_values_list)
                    ]

                case_outputs[df_label] = filtered_df

            # Store the results for the current case
            outputs[case_label] = case_outputs

        return outputs

    @staticmethod
    def apply_post_filter_funcs(
        metafilter_result: MetaFilterResult,
        post_filter_funcs: PostFilterFuncDict,
    ) -> PostFilterFuncResult:
        """
        Apply post-filter functions to the results obtained from `metafilter_df`.

        Iterates through the nested dictionary structure produced by `metafilter_df`.
        For each filtered DataFrame within each case, it checks if a corresponding
        function exists in the `post_filter_funcs` dictionary based on the DataFrame's
        original label (`dflabel`). If a function is found, it is applied to a *copy*
        of that filtered DataFrame. The output of the function replaces the original
        filtered DataFrame in the results structure for that specific case and dflabel.

        Note: The functions provided in `post_filter_funcs` receive the *filtered DataFrame*
        as input. They can return any value or structure, which will
        then be stored under the corresponding `dflabel` key within the case.

        Args:
            metafilter_result (MetaFilterResult): The nested dictionary output from
                `metafilter_df`. Structure: {case_label: {df_label: filtered_dataframe}}
            post_filter_funcs (PostFilterFuncDict): A dictionary mapping DataFrame
                labels (str) to functions. Each function is applied to a copy of the
                filtered DataFrame corresponding to its key within each case. The function
                must accept a pandas DataFrame (the filtered DataFrame) as input.

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
            >>> # Continuing from the metafilter_df example...
            >>> import pandas as pd
            >>> import collections.abc
            >>> from typing import Dict, Union, TypeAlias, Callable, Any # For type hints in example
            >>> FilterValue: TypeAlias = Union[str, int, collections.abc.Iterable]
            >>> FilterDict: TypeAlias = Dict[str, FilterValue]
            >>> DataFrameDict: TypeAlias = Dict[str, pd.DataFrame]
            >>> CaseFilterDict: TypeAlias = Dict[str, FilterDict]
            >>> MetaFilterConfig: TypeAlias = Dict[str, CaseFilterDict]
            >>> MetaFilterResult: TypeAlias = Dict[str, Dict[str, pd.DataFrame]]
            >>> PostFilterFuncDict: TypeAlias = Dict[str, Callable[[pd.DataFrame], Any]] # Note: Return type Any
            >>> PostFilterFuncResult: TypeAlias = Dict[str, Dict[str, Any]]
            >>>
            >>> # Example DataFrames (same as metafilter_df example)
            >>> df_sales = pd.DataFrame({
            ...     'Region': ['North', 'South', 'North', 'West'],
            ...     'Sales': [100, 150, 200, 50], 'Units': [10, 15, 20, 5]
            ... })
            >>> df_customers = pd.DataFrame({
            ...     'CustomerID': [1, 2, 3, 4], 'Region': ['South', 'North', 'West', 'North'],
            ...     'Segment': ['A', 'B', 'A', 'B']
            ... })
            >>> dfs_input: DataFrameDict = {"sales": df_sales, "customers": df_customers}
            >>>
            >>> # Example Filter Configuration (same as metafilter_df example)
            >>> filter_config: MetaFilterConfig = {
            ...     "North_Region": {"sales": {"Region": "North"}, "customers": {"Region": "North"}},
            ...     "High_Sales_West_Customers": {"sales": {"Sales": [150, 200]}, "customers": {"Region": "West"}}
            ... }
            >>>
            >>> # First, get results from metafilter_df
            >>> initial_results = Metafilter.metafilter_df(dfs_input, filter_config)
            >>>
            >>> # Define a post-filter function for the 'sales' DataFrame
            >>> # This function takes the *filtered* DataFrame and returns a count of rows
            >>> def count_filtered_sales(filtered_df: pd.DataFrame) -> str:
            ...     return f"Sales Count: {len(filtered_df)}"
            >>>
            >>> # Define another function for 'customers'
            >>> def get_customer_ids(filtered_df: pd.DataFrame) -> list:
            ...     return filtered_df['CustomerID'].tolist()
            >>>
            >>> # Define the dictionary linking dflabels to the functions
            >>> post_funcs: PostFilterFuncDict = {
            ...     "sales": count_filtered_sales,
            ...     "customers": get_customer_ids
            ... }
            >>>
            >>> # Apply the post-filter functions
            >>> final_results = Metafilter.apply_post_filter_funcs(initial_results, post_funcs)
            >>>
            >>> # Check the results - 'sales' and 'customers' entries are now function outputs
            >>> print("--- Case: North_Region (Post-Processed) ---")
            >>> print(final_results["North_Region"]["sales"]) # Output is a string
            >>> print(final_results["North_Region"]["customers"]) # Output is a list
            >>>
            >>> print("\n--- Case: High_Sales_West_Customers (Post-Processed) ---")
            >>> print(final_results["High_Sales_West_Customers"]["sales"]) # Output is a string
            >>> print(final_results["High_Sales_West_Customers"]["customers"]) # Output is a list
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
