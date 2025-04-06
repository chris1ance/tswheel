import pytest
import pandas as pd
import numpy as np
from tswheel.datautils.pdutils import merge_on_index


class TestMergeOnIndex:
    """Tests for the merge_on_index function."""

    def test_inner_merge(self):
        """Test inner merge of two DataFrames with partially overlapping indices."""
        # Create DataFrames with overlapping DatetimeIndex
        df1 = pd.DataFrame(
            {"a": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame(
            {"b": [4, 5, 6]}, index=pd.date_range("2022-01-02", periods=3, freq="D")
        )

        # Expected result: only overlapping dates (Jan 2-3)
        expected_index = pd.date_range("2022-01-02", periods=2, freq="D")
        expected_df = pd.DataFrame({"a": [2, 3], "b": [4, 5]}, index=expected_index)

        result = merge_on_index(df1, df2)

        pd.testing.assert_frame_equal(result, expected_df)

    def test_outer_merge(self):
        """Test outer merge of two DataFrames with partially overlapping indices."""
        # Create DataFrames with overlapping DatetimeIndex
        df1 = pd.DataFrame(
            {"a": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame(
            {"b": [4, 5, 6]}, index=pd.date_range("2022-01-02", periods=3, freq="D")
        )

        # Expected result: union of all dates with NaN where data is missing
        expected_index = pd.date_range("2022-01-01", periods=4, freq="D")
        expected_data = {"a": [1.0, 2.0, 3.0, np.nan], "b": [np.nan, 4.0, 5.0, 6.0]}
        expected_df = pd.DataFrame(expected_data, index=expected_index)

        result = merge_on_index(df1, df2, how="outer")

        pd.testing.assert_frame_equal(result, expected_df)

    def test_left_merge(self):
        """Test left merge of two DataFrames with partially overlapping indices."""
        # Create DataFrames with overlapping DatetimeIndex
        df1 = pd.DataFrame(
            {"a": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame(
            {"b": [4, 5, 6]}, index=pd.date_range("2022-01-02", periods=3, freq="D")
        )

        # Expected result: all dates from df1 with NaN where df2 data is missing
        expected_index = pd.date_range("2022-01-01", periods=3, freq="D")
        expected_data = {"a": [1, 2, 3], "b": [np.nan, 4.0, 5.0]}
        expected_df = pd.DataFrame(expected_data, index=expected_index)

        result = merge_on_index(df1, df2, how="left")

        pd.testing.assert_frame_equal(result, expected_df)

    def test_right_merge(self):
        """Test right merge of two DataFrames with partially overlapping indices."""
        # Create DataFrames with overlapping DatetimeIndex
        df1 = pd.DataFrame(
            {"a": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame(
            {"b": [4, 5, 6]}, index=pd.date_range("2022-01-02", periods=3, freq="D")
        )

        # Expected result: all dates from df2 with NaN where df1 data is missing
        expected_index = pd.date_range("2022-01-02", periods=3, freq="D")
        expected_data = {"a": [2.0, 3.0, np.nan], "b": [4, 5, 6]}
        expected_df = pd.DataFrame(expected_data, index=expected_index)

        result = merge_on_index(df1, df2, how="right")

        pd.testing.assert_frame_equal(result, expected_df)

    def test_incompatible_indices(self):
        """Test that an error is raised when merging DataFrames with incompatible indices."""
        # Create DataFrames with different index types
        df1 = pd.DataFrame(
            {"a": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame({"b": [4, 5, 6]}, index=pd.RangeIndex(start=0, stop=3))

        with pytest.raises(ValueError) as excinfo:
            merge_on_index(df1, df2)

        assert "Index types differ" in str(excinfo.value)

    def test_custom_suffixes(self):
        """Test merge with custom suffixes for overlapping column names."""
        # Create DataFrames with overlapping column names
        df1 = pd.DataFrame(
            {"value": [1, 2, 3], "id": ["A", "B", "C"]},
            index=pd.date_range("2022-01-01", periods=3, freq="D"),
        )
        df2 = pd.DataFrame(
            {"value": [4, 5, 6], "code": [100, 200, 300]},
            index=pd.date_range("2022-01-02", periods=3, freq="D"),
        )

        # Expected result with custom suffixes
        expected_index = pd.date_range("2022-01-02", periods=2, freq="D")
        expected_data = {
            "value_left": [2, 3],
            "id": ["B", "C"],
            "value_right": [4, 5],
            "code": [100, 200],
        }
        expected_df = pd.DataFrame(expected_data, index=expected_index)

        result = merge_on_index(df1, df2, suffixes=("_left", "_right"))

        pd.testing.assert_frame_equal(result, expected_df)

    def test_with_indicator(self):
        """Test merge with indicator column."""
        # Create DataFrames with overlapping DatetimeIndex
        df1 = pd.DataFrame(
            {"a": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame(
            {"b": [4, 5, 6]}, index=pd.date_range("2022-01-02", periods=3, freq="D")
        )

        result = merge_on_index(df1, df2, how="outer", indicator=True)

        # Check indicator column exists and has correct values
        assert "_merge" in result.columns
        assert result.loc[pd.Timestamp("2022-01-01"), "_merge"] == "left_only"
        assert result.loc[pd.Timestamp("2022-01-02"), "_merge"] == "both"
        assert result.loc[pd.Timestamp("2022-01-03"), "_merge"] == "both"
        assert result.loc[pd.Timestamp("2022-01-04"), "_merge"] == "right_only"

    def test_with_validate(self):
        """Test merge with validation."""
        # Create 1:1 matching DataFrames
        df1 = pd.DataFrame(
            {"a": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame(
            {"b": [4, 5, 6]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )

        # This should work with 1:1 validation
        result = merge_on_index(df1, df2, validate="1:1")
        assert len(result) == 3

        # Create non-1:1 matching DataFrames (with duplicate indices)
        df3 = pd.DataFrame(
            {"a": [1, 2, 3, 4]},
            index=pd.DatetimeIndex(
                ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-01"]
            ),
        )

        # This should raise a validation error first before frequency check
        with pytest.raises(ValueError) as excinfo:
            merge_on_index(df3, df2, validate="1:1")

        assert "contains duplicate entries" in str(excinfo.value)
