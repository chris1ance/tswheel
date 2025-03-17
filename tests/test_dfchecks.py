import pandas as pd
from tswheel.datawork.dfchecks import have_same_index_type


class TestHaveSameIndexType:
    """Tests for the have_same_index_type function."""

    def test_same_index_type_and_frequency(self):
        """Test when both DataFrames have the same index type and frequency."""
        # Create DataFrames with DatetimeIndex and same frequency
        df1 = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame(
            {"B": [4, 5, 6]}, index=pd.date_range("2022-02-01", periods=3, freq="D")
        )

        result, message = have_same_index_type(df1, df2)

        assert result is True
        assert "same type and frequency" in message

    def test_same_index_type_different_frequency(self):
        """Test when DataFrames have same index type but different frequencies."""
        # Create DataFrames with DatetimeIndex but different frequencies
        df1 = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame(
            {"B": [4, 5, 6]}, index=pd.date_range("2022-01-01", periods=3, freq="ME")
        )

        result, message = have_same_index_type(df1, df2)

        assert result is False
        assert "different frequencies" in message
        assert "<Day> vs <MonthEnd>" in message

    def test_different_index_types(self):
        """Test when DataFrames have different index types."""
        # Create DataFrames with different index types
        df1 = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3)
        )
        df2 = pd.DataFrame({"B": [4, 5, 6]}, index=pd.RangeIndex(start=0, stop=3))

        result, message = have_same_index_type(df1, df2)

        assert result is False
        assert "Index types differ" in message
        assert "DatetimeIndex vs RangeIndex" in message

    def test_both_none_frequency(self):
        """Test when both DataFrames have time-based indices with None frequency."""
        # Create DataFrames with DatetimeIndex but None frequency
        df1 = pd.DataFrame(
            {"A": [1, 2, 3]},
            index=pd.DatetimeIndex(["2022-01-01", "2022-01-02", "2022-01-03"]),
        )
        df2 = pd.DataFrame(
            {"B": [4, 5, 6]},
            index=pd.DatetimeIndex(["2022-02-01", "2022-02-02", "2022-02-03"]),
        )

        result, message = have_same_index_type(df1, df2)

        assert result is True
        assert "undefined frequency" in message

    def test_one_none_frequency(self):
        """Test when one DataFrame has a frequency and the other doesn't."""
        # Create one DataFrame with frequency and one without
        df1 = pd.DataFrame(
            {"A": [1, 2, 3]}, index=pd.date_range("2022-01-01", periods=3, freq="D")
        )
        df2 = pd.DataFrame(
            {"B": [4, 5, 6]},
            index=pd.DatetimeIndex(["2022-02-01", "2022-02-02", "2022-02-03"]),
        )

        result, message = have_same_index_type(df1, df2)

        assert result is False
        assert "one has undefined frequency" in message

    def test_non_timebased_indices(self):
        """Test when DataFrames have indices without frequency attribute."""
        # Create DataFrames with non-time-based indices
        df1 = pd.DataFrame({"A": [1, 2, 3]}, index=pd.RangeIndex(start=0, stop=3))
        df2 = pd.DataFrame({"B": [4, 5, 6]}, index=pd.RangeIndex(start=3, stop=6))

        result, message = have_same_index_type(df1, df2)

        assert result is True
        assert "same type" in message

    def test_multi_index(self):
        """Test when DataFrames have MultiIndex."""
        # Create DataFrames with MultiIndex
        df1 = pd.DataFrame(
            {"A": [1, 2, 3, 4]},
            index=pd.MultiIndex.from_tuples(
                [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["letter", "number"]
            ),
        )
        df2 = pd.DataFrame(
            {"B": [5, 6, 7, 8]},
            index=pd.MultiIndex.from_tuples(
                [("c", 3), ("c", 4), ("d", 3), ("d", 4)], names=["letter", "number"]
            ),
        )

        result, message = have_same_index_type(df1, df2)

        assert result is True
        assert "same type" in message
