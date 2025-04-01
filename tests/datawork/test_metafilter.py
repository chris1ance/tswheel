import pytest
import pandas as pd

from tswheel.datawork.metafilter import (
    Metafilter,
    FilterDict,
    DataFrameDict,
    MetaFilterConfig,
    PostFilterFuncDict,
    MetaFilterResult,
)


class TestFilterDF:
    """Tests for the filter_df method of Metafilter class."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing filter_df."""
        return pd.DataFrame(
            {
                "category": ["A", "B", "A", "C", "B", "A"],
                "region": ["North", "South", "North", "East", "South", "West"],
                "value": [10, 20, 30, 40, 50, 60],
                "active": [True, True, False, True, False, True],
            }
        )

    def test_filter_single_value(self, sample_df):
        """Test filtering with a single value for a column."""
        # Filter for category 'A'
        filters: FilterDict = {"category": "A"}
        result = Metafilter.filter_df(sample_df, filters, label="CategoryA")

        # Check summary DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert "category" in result.columns
        assert "result" in result.columns
        assert result.index[0] == "CategoryA"

        # Check filtered DataFrame in the result column
        filtered = result.loc["CategoryA", "result"]
        assert len(filtered) == 3  # Should have 3 rows with category 'A'
        assert all(filtered["category"] == "A")
        assert set(filtered.index) == {0, 2, 5}  # Original indices should be preserved

    def test_filter_multiple_values(self, sample_df):
        """Test filtering with multiple values for a column."""
        # Filter for categories 'A' and 'B'
        filters: FilterDict = {"category": ["A", "B"]}
        result = Metafilter.filter_df(sample_df, filters, label="CategoryAB")

        filtered = result.loc["CategoryAB", "result"]
        assert len(filtered) == 5  # Should have 5 rows with category 'A' or 'B'
        assert all(filtered["category"].isin(["A", "B"]))

    def test_filter_multiple_columns(self, sample_df):
        """Test filtering on multiple columns simultaneously."""
        # Filter for category 'A' and region 'North'
        filters: FilterDict = {"category": "A", "region": "North"}
        result = Metafilter.filter_df(sample_df, filters, label="ANorth")

        filtered = result.loc["ANorth", "result"]
        assert len(filtered) == 2  # Should have 2 rows matching both criteria
        assert all(filtered["category"] == "A")
        assert all(filtered["region"] == "North")
        assert set(filtered.index) == {0, 2}

    def test_filter_empty_dict(self, sample_df):
        """Test behavior with empty filters dictionary."""
        # Empty filters should return original DataFrame
        result = Metafilter.filter_df(sample_df, {})
        assert result.equals(sample_df)  # Should return original DataFrame unchanged

    def test_filter_boolean_column(self, sample_df):
        """Test filtering on a boolean column."""
        # Filter for active=True
        filters: FilterDict = {"active": True}
        result = Metafilter.filter_df(sample_df, filters, label="Active")

        filtered = result.loc["Active", "result"]
        assert len(filtered) == 4  # Should have 4 rows with active=True
        assert all(filtered["active"])

    def test_filter_numeric_column(self, sample_df):
        """Test filtering on a numeric column."""
        # Filter for values >= 30
        filters: FilterDict = {"value": [30, 40, 50, 60]}
        result = Metafilter.filter_df(sample_df, filters, label="HighValue")

        filtered = result.loc["HighValue", "result"]
        assert len(filtered) == 4  # Should have 4 rows with value >= 30
        assert all(filtered["value"] >= 30)

    def test_error_column_not_found(self, sample_df):
        """Test error when column doesn't exist in DataFrame."""
        # Column 'nonexistent' doesn't exist
        filters: FilterDict = {"nonexistent": "value"}
        with pytest.raises(ValueError) as excinfo:
            Metafilter.filter_df(sample_df, filters)
        assert "Column 'nonexistent' not found in DataFrame" in str(excinfo.value)

    def test_error_empty_result(self, sample_df):
        """Test error when filters result in empty DataFrame."""
        # No rows have category 'X'
        filters: FilterDict = {"category": "X"}
        with pytest.raises(ValueError) as excinfo:
            Metafilter.filter_df(sample_df, filters)
        assert "empty DataFrame" in str(excinfo.value)

    def test_error_invalid_filter_value_type(self, sample_df):
        """Test error with invalid filter value type."""
        # Use a non-iterable, non-scalar value (None) which should cause a TypeError
        filters: FilterDict = {"category": None}
        with pytest.raises(TypeError) as excinfo:
            Metafilter.filter_df(sample_df, filters)
        assert "must be a list-like iterable" in str(excinfo.value)

    def test_error_invalid_df_type(self):
        """Test error when df is not a pandas DataFrame."""
        # df parameter must be a DataFrame
        with pytest.raises(TypeError) as excinfo:
            Metafilter.filter_df("not_a_dataframe", {"col": "val"})
        assert "must be a pandas DataFrame" in str(excinfo.value)

    def test_error_invalid_filters_type(self, sample_df):
        """Test error when filters is not a dictionary."""
        # filters parameter must be a dictionary
        with pytest.raises(TypeError) as excinfo:
            Metafilter.filter_df(sample_df, "not_a_dict")
        assert "must be a dictionary" in str(excinfo.value)

    def test_error_invalid_label_type(self, sample_df):
        """Test error when label is not a string."""
        # label parameter must be a string or None
        with pytest.raises(TypeError) as excinfo:
            Metafilter.filter_df(sample_df, {"category": "A"}, label=123)
        assert "must be a string or None" in str(excinfo.value)


class TestMetaFilterDF:
    """Tests for the metafilter_df method of Metafilter class."""

    @pytest.fixture
    def sample_dfs(self) -> DataFrameDict:
        """Create sample DataFrames for testing metafilter_df."""
        # Sales DataFrame
        df_sales = pd.DataFrame(
            {
                "product": ["X", "Y", "Z", "X", "Y"],
                "region": ["North", "South", "North", "East", "West"],
                "sales": [1000, 2000, 1500, 3000, 2500],
                "quarter": ["Q1", "Q1", "Q2", "Q2", "Q3"],
            }
        )

        # Customers DataFrame
        df_customers = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "region": ["North", "South", "East", "West", "North"],
                "segment": ["A", "B", "A", "B", "C"],
                "active": [True, True, False, True, False],
            }
        )

        return {"sales": df_sales, "customers": df_customers}

    def test_basic_filtering(self, sample_dfs):
        """Test basic filtering across multiple DataFrames."""
        # Define filters for one case, applying to both DataFrames
        filters: MetaFilterConfig = {
            "North_Region": {
                "sales": {"region": "North"},
                "customers": {"region": "North"},
            }
        }

        result = Metafilter.metafilter_df(sample_dfs, filters)

        # Check structure of the result
        assert "North_Region" in result
        assert "sales" in result["North_Region"]
        assert "customers" in result["North_Region"]

        # Check filtered sales DataFrame
        sales_filtered = result["North_Region"]["sales"].loc["North_Region", "result"]
        assert len(sales_filtered) == 2  # Should have 2 rows with region 'North'
        assert all(sales_filtered["region"] == "North")

        # Check filtered customers DataFrame
        customers_filtered = result["North_Region"]["customers"].loc[
            "North_Region", "result"
        ]
        assert len(customers_filtered) == 2  # Should have 2 rows with region 'North'
        assert all(customers_filtered["region"] == "North")

    def test_multiple_cases(self, sample_dfs):
        """Test filtering with multiple cases."""
        # Define filters for two cases
        filters: MetaFilterConfig = {
            "North_Region": {
                "sales": {"region": "North"},
                "customers": {"region": "North"},
            },
            "High_Sales": {
                "sales": {"sales": [2000, 2500, 3000]},  # Sales >= 2000
            },
        }

        result = Metafilter.metafilter_df(sample_dfs, filters)

        # Check North_Region case
        assert "North_Region" in result
        north_sales = result["North_Region"]["sales"].loc["North_Region", "result"]
        assert len(north_sales) == 2

        # Check High_Sales case
        assert "High_Sales" in result
        high_sales = result["High_Sales"]["sales"].loc["High_Sales", "result"]
        assert len(high_sales) == 3  # Should have 3 rows with sales >= 2000
        assert all(high_sales["sales"] >= 2000)

        # High_Sales case should only have 'sales' DataFrame filtered
        assert "customers" not in result["High_Sales"]

    def test_complex_filters(self, sample_dfs):
        """Test with complex filtering criteria."""
        # Define complex filters combining multiple conditions
        filters: MetaFilterConfig = {
            "NorthRegion_SegmentA": {
                "sales": {"region": "North", "quarter": ["Q1", "Q2"]},
                "customers": {"region": "North", "segment": "A"},
            }
        }

        result = Metafilter.metafilter_df(sample_dfs, filters)

        # Check filtered sales DataFrame (North region AND Q1/Q2)
        sales_filtered = result["NorthRegion_SegmentA"]["sales"].loc[
            "NorthRegion_SegmentA", "result"
        ]
        assert len(sales_filtered) == 2
        assert all(sales_filtered["region"] == "North")
        assert all(sales_filtered["quarter"].isin(["Q1", "Q2"]))

        # Check filtered customers DataFrame (North region AND segment A)
        customers_filtered = result["NorthRegion_SegmentA"]["customers"].loc[
            "NorthRegion_SegmentA", "result"
        ]
        assert len(customers_filtered) == 1
        assert all(customers_filtered["region"] == "North")
        assert all(customers_filtered["segment"] == "A")

    def test_empty_filters(self, sample_dfs):
        """Test behavior with empty filters dictionary."""
        # Empty filters should return empty result
        result = Metafilter.metafilter_df(sample_dfs, {})
        assert result == {}

    def test_error_df_label_not_found(self, sample_dfs):
        """Test error when DataFrame label is not found."""
        # DataFrame label 'nonexistent' doesn't exist
        filters: MetaFilterConfig = {"TestCase": {"nonexistent": {"col": "val"}}}
        with pytest.raises(ValueError) as excinfo:
            Metafilter.metafilter_df(sample_dfs, filters)
        assert "DataFrame label 'nonexistent'" in str(excinfo.value)

    def test_error_invalid_dfs_type(self):
        """Test error when dfs is not a dictionary."""
        # dfs parameter must be a dictionary
        with pytest.raises(TypeError) as excinfo:
            Metafilter.metafilter_df("not_a_dict", {"case": {"df": {"col": "val"}}})
        assert "must be a dictionary" in str(excinfo.value)

    def test_error_invalid_filters_type(self, sample_dfs):
        """Test error when filters is not a dictionary."""
        # filters parameter must be a dictionary
        with pytest.raises(TypeError) as excinfo:
            Metafilter.metafilter_df(sample_dfs, "not_a_dict")
        assert "must be a dictionary" in str(excinfo.value)

    def test_error_invalid_df_object(self):
        """Test error when a value in dfs is not a DataFrame."""
        # Values in dfs must be DataFrames
        invalid_dfs = {"df1": "not_a_dataframe"}
        with pytest.raises(TypeError) as excinfo:
            Metafilter.metafilter_df(invalid_dfs, {"case": {"df1": {"col": "val"}}})
        assert "must be a pandas DataFrame" in str(excinfo.value)

    def test_error_propagation_from_filter_df(self, sample_dfs):
        """Test error propagation from filter_df function."""
        # This will cause filter_df to raise an error (empty result)
        filters: MetaFilterConfig = {"TestCase": {"sales": {"product": "NonExistent"}}}
        with pytest.raises(ValueError) as excinfo:
            Metafilter.metafilter_df(sample_dfs, filters)
        assert "empty DataFrame" in str(excinfo.value)


class TestApplyPostFilterFuncs:
    """Tests for the apply_post_filter_funcs method of Metafilter class."""

    @pytest.fixture
    def sample_dfs(self) -> DataFrameDict:
        """Create sample DataFrames for testing."""
        # Sales DataFrame
        df_sales = pd.DataFrame(
            {
                "product": ["X", "Y", "Z", "X", "Y"],
                "region": ["North", "South", "North", "East", "West"],
                "sales": [1000, 2000, 1500, 3000, 2500],
                "quarter": ["Q1", "Q1", "Q2", "Q2", "Q3"],
            }
        )

        # Customers DataFrame
        df_customers = pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "region": ["North", "South", "East", "West", "North"],
                "segment": ["A", "B", "A", "B", "C"],
                "active": [True, True, False, True, False],
            }
        )

        return {"sales": df_sales, "customers": df_customers}

    @pytest.fixture
    def metafilter_result(self, sample_dfs) -> MetaFilterResult:
        """Create a sample metafilter result for testing."""
        filters: MetaFilterConfig = {
            "North_Region": {
                "sales": {"region": "North"},
                "customers": {"region": "North"},
            }
        }
        return Metafilter.metafilter_df(sample_dfs, filters)

    def test_basic_functionality(self, metafilter_result):
        """Test basic functionality of apply_post_filter_funcs."""

        # Define a simple post-filter function
        def add_count_column(summary_df: pd.DataFrame) -> pd.DataFrame:
            # Create a new DataFrame with summary info
            case_label = summary_df.index[0]
            filtered_df = summary_df.loc[case_label, "result"]
            return pd.DataFrame({"case": [case_label], "row_count": [len(filtered_df)]})

        # Define post-filter functions
        post_funcs: PostFilterFuncDict = {"sales": add_count_column}

        # Apply post-filter functions
        result = Metafilter.apply_post_filter_funcs(metafilter_result, post_funcs)

        # Check that post-filter function was applied
        post_result = result["North_Region"]["sales"]

        # Verify post-filter result
        assert isinstance(post_result, pd.DataFrame)
        assert "case" in post_result.columns
        assert "row_count" in post_result.columns
        assert post_result.iloc[0]["case"] == "North_Region"
        assert (
            post_result.iloc[0]["row_count"] == 2
        )  # Should be 2 rows with region 'North'

    def test_multiple_functions(self, metafilter_result):
        """Test apply_post_filter_funcs with multiple DataFrames and functions."""

        # Define post-filter functions for different DataFrames
        def summarize_sales(summary_df: pd.DataFrame) -> pd.DataFrame:
            case_label = summary_df.index[0]
            filtered_df = summary_df.loc[case_label, "result"]
            return pd.DataFrame(
                {
                    "total_sales": [filtered_df["sales"].sum()],
                    "avg_sales": [filtered_df["sales"].mean()],
                }
            )

        def summarize_customers(summary_df: pd.DataFrame) -> pd.DataFrame:
            case_label = summary_df.index[0]
            filtered_df = summary_df.loc[case_label, "result"]
            return pd.DataFrame(
                {
                    "active_count": [filtered_df["active"].sum()],
                    "total_count": [len(filtered_df)],
                }
            )

        # Define post-filter functions
        post_funcs: PostFilterFuncDict = {
            "sales": summarize_sales,
            "customers": summarize_customers,
        }

        # Apply post-filter functions
        result = Metafilter.apply_post_filter_funcs(metafilter_result, post_funcs)

        # Check sales post-filter result
        sales_post = result["North_Region"]["sales"]
        assert "total_sales" in sales_post.columns
        assert "avg_sales" in sales_post.columns

        # Check customers post-filter result
        customers_post = result["North_Region"]["customers"]
        assert "active_count" in customers_post.columns
        assert "total_count" in customers_post.columns

        # Verify specific values
        assert (
            sales_post.iloc[0]["total_sales"] == 2500
        )  # Sum of sales for North region
        assert customers_post.iloc[0]["total_count"] == 2  # 2 customers in North region

    def test_partial_application(self, metafilter_result):
        """Test apply_post_filter_funcs with function for only some DataFrames."""

        # Define post-filter function only for sales
        def summarize_sales(summary_df: pd.DataFrame) -> pd.DataFrame:
            case_label = summary_df.index[0]
            filtered_df = summary_df.loc[case_label, "result"]
            return pd.DataFrame({"total_sales": [filtered_df["sales"].sum()]})

        # Define post-filter only for sales
        post_funcs: PostFilterFuncDict = {"sales": summarize_sales}

        # Apply post-filter functions
        result = Metafilter.apply_post_filter_funcs(metafilter_result, post_funcs)

        # Check that post-filter was applied only to sales
        assert isinstance(result["North_Region"]["sales"], pd.DataFrame)
        assert "total_sales" in result["North_Region"]["sales"].columns

        # Customers should still be the original DataFrame
        assert "result" in result["North_Region"]["customers"].columns

    def test_error_invalid_result_type(self):
        """Test error when metafilter_result is not a dictionary."""
        post_funcs: PostFilterFuncDict = {"sales": lambda df: df}

        with pytest.raises(TypeError) as excinfo:
            Metafilter.apply_post_filter_funcs("not_a_dict", post_funcs)
        assert "must be a dictionary" in str(excinfo.value)

    def test_error_invalid_post_funcs_type(self, metafilter_result):
        """Test error when post_filter_funcs is not a dictionary."""
        with pytest.raises(TypeError) as excinfo:
            Metafilter.apply_post_filter_funcs(metafilter_result, "not_a_dict")
        assert "must be a dictionary" in str(excinfo.value)

    def test_error_post_func_not_callable(self, metafilter_result):
        """Test error when a post-filter function is not callable."""
        post_funcs: PostFilterFuncDict = {"sales": "not_callable"}

        with pytest.raises(TypeError) as excinfo:
            Metafilter.apply_post_filter_funcs(metafilter_result, post_funcs)
        assert "must be callable" in str(excinfo.value)

    def test_error_propagation_from_post_func(self, metafilter_result):
        """Test error propagation from post-filter function."""

        def failing_func(_: pd.DataFrame) -> pd.DataFrame:
            raise ValueError("Intentional error in post-filter function")

        post_funcs: PostFilterFuncDict = {"sales": failing_func}

        with pytest.raises(ValueError) as excinfo:
            Metafilter.apply_post_filter_funcs(metafilter_result, post_funcs)
        assert "Intentional error in post-filter function" in str(excinfo.value)
