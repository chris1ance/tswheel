import pytest
import pandas as pd
import numpy as np

from tswheel.datawork.dfutils import ts_concat


def test_basic_row_concatenation() -> None:
    """
    Test basic row concatenation (axis=0) with DataFrames.
    """
    # --- Setup ---
    df1 = pd.DataFrame({"A": [1, 2]}, index=pd.RangeIndex(2))
    df2 = pd.DataFrame({"A": [3, 4]}, index=pd.RangeIndex(start=2, stop=4))

    # --- Execute ---
    result = ts_concat([df1, df2])

    # --- Assert ---
    expected = pd.DataFrame({"A": [1, 2, 3, 4]}, index=pd.RangeIndex(4))
    pd.testing.assert_frame_equal(result, expected)


def test_basic_column_concatenation() -> None:
    """
    Test basic column concatenation (axis=1) with DataFrames.
    """
    # --- Setup ---
    df1 = pd.DataFrame({"A": [1, 2]}, index=pd.RangeIndex(2))
    df2 = pd.DataFrame({"B": [3, 4]}, index=pd.RangeIndex(2))

    # --- Execute ---
    result = ts_concat([df1, df2], axis=1)

    # --- Assert ---
    expected = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=pd.RangeIndex(2))
    pd.testing.assert_frame_equal(result, expected)


def test_series_concatenation() -> None:
    """
    Test concatenation of Series objects.
    """
    # --- Setup ---
    s1 = pd.Series([1, 2], index=pd.RangeIndex(2))
    s2 = pd.Series([3, 4], index=pd.RangeIndex(start=2, stop=4))

    # --- Execute ---
    result = ts_concat([s1, s2])

    # --- Assert ---
    expected = pd.Series([1, 2, 3, 4], index=pd.RangeIndex(4))
    pd.testing.assert_series_equal(result, expected)


def test_mixed_series_df_concatenation() -> None:
    """
    Test concatenation of mixed Series and DataFrame objects.
    """
    # --- Setup ---
    s1 = pd.Series([1, 2], index=pd.RangeIndex(2), name="A")
    df1 = pd.DataFrame({"B": [3, 4]}, index=pd.RangeIndex(2))

    # --- Execute ---
    result = ts_concat([s1, df1], axis=1)

    # --- Assert ---
    expected = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=pd.RangeIndex(2))
    pd.testing.assert_frame_equal(result, expected)


def test_join_inner() -> None:
    """
    Test concatenation with inner join.
    """
    # --- Setup ---
    df1 = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({"B": [3, 4]}, index=[1, 2])

    # --- Execute ---
    result = ts_concat([df1, df2], join="inner")

    # --- Assert ---
    # With inner join, we keep only columns in all dataframes (none in this case)
    # Create empty DataFrame with correct index for comparison
    expected = pd.DataFrame({}, index=[0, 1, 1, 2])
    pd.testing.assert_frame_equal(
        result, expected, check_index_type=True, check_column_type=False
    )


def test_join_outer() -> None:
    """
    Test concatenation with outer join.
    """
    # --- Setup ---
    df1 = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({"B": [3, 4]}, index=[1, 2])

    # --- Execute ---
    result = ts_concat([df1, df2], join="outer")

    # --- Assert ---
    # With outer join, all columns are kept
    expected = pd.DataFrame(
        {"A": [1.0, 2.0, np.nan, np.nan], "B": [np.nan, np.nan, 3.0, 4.0]},
        index=[0, 1, 1, 2],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_ignore_index() -> None:
    """
    Test concatenation with ignore_index=True.
    """
    # --- Setup ---
    df1 = pd.DataFrame({"A": [1, 2]}, index=["a", "b"])
    df2 = pd.DataFrame({"A": [3, 4]}, index=["c", "d"])

    # --- Execute ---
    result = ts_concat([df1, df2], ignore_index=True)

    # --- Assert ---
    expected = pd.DataFrame({"A": [1, 2, 3, 4]}, index=pd.RangeIndex(4))
    pd.testing.assert_frame_equal(result, expected)


def test_datetimeindex_concatenation() -> None:
    """
    Test concatenation with DatetimeIndex.
    """
    # --- Setup ---
    df1 = pd.DataFrame(
        {"A": [1, 2]}, index=pd.date_range("2022-01-01", periods=2, freq="D")
    )
    df2 = pd.DataFrame(
        {"A": [3, 4]}, index=pd.date_range("2022-01-03", periods=2, freq="D")
    )

    # --- Execute ---
    result = ts_concat([df1, df2])

    # --- Assert ---
    expected = pd.DataFrame(
        {"A": [1, 2, 3, 4]}, index=pd.date_range("2022-01-01", periods=4, freq="D")
    )
    pd.testing.assert_frame_equal(result, expected)


def test_periodindex_concatenation() -> None:
    """
    Test concatenation with PeriodIndex.
    """
    # --- Setup ---
    df1 = pd.DataFrame(
        {"A": [1, 2]}, index=pd.period_range("2022-01", periods=2, freq="M")
    )
    df2 = pd.DataFrame(
        {"A": [3, 4]}, index=pd.period_range("2022-03", periods=2, freq="M")
    )

    # --- Execute ---
    result = ts_concat([df1, df2])

    # --- Assert ---
    expected = pd.DataFrame(
        {"A": [1, 2, 3, 4]}, index=pd.period_range("2022-01", periods=4, freq="M")
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "test_input,expected_error",
    [
        # Not a list
        ((pd.DataFrame({"A": [1]}), pd.DataFrame({"B": [2]})), TypeError),
        # Empty list
        ([], ValueError),
        # Non-pandas object
        ([pd.DataFrame({"A": [1]}), [1, 2]], TypeError),
    ],
)
def test_invalid_input_types(test_input: tuple, expected_error: type) -> None:
    """
    Test that appropriate errors are raised for invalid input types.

    Args:
        test_input (tuple): The input to pass to ts_concat
        expected_error (type): The expected error type
    """
    # --- Assert ---
    with pytest.raises(expected_error):
        ts_concat(test_input)


def test_error_on_different_index_types() -> None:
    """
    Test that an error is raised when concatenating objects with different index types.
    """
    # --- Setup ---
    df1 = pd.DataFrame({"A": [1, 2]}, index=pd.RangeIndex(2))
    df2 = pd.DataFrame({"B": [3, 4]}, index=pd.Index(["x", "y"]))

    # --- Assert ---
    with pytest.raises(ValueError) as excinfo:
        ts_concat([df1, df2])
    assert "Index types differ" in str(excinfo.value)


def test_error_on_duplicate_indices() -> None:
    """
    Test that an error is raised when concatenating objects with duplicate indices.
    """
    # --- Setup ---
    df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 1])  # Duplicate index
    df2 = pd.DataFrame({"B": [4, 5]}, index=[2, 3])

    # --- Assert ---
    with pytest.raises(ValueError) as excinfo:
        ts_concat([df1, df2])
    assert "duplicate index values" in str(excinfo.value)


def test_error_on_duplicate_columns() -> None:
    """
    Test that an error is raised when concatenating along columns with duplicate column names.
    """
    # --- Setup ---
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5, 6], "C": [7, 8]})  # 'A' column exists in both

    # --- Assert ---
    with pytest.raises(ValueError) as excinfo:
        ts_concat([df1, df2], axis=1)
    assert "Duplicate column names" in str(excinfo.value)


def test_error_on_unnamed_series_column_concat() -> None:
    """
    Test that an error is raised when concatenating unnamed Series along columns.
    """
    # --- Setup ---
    df = pd.DataFrame({"A": [1, 2]})
    s = pd.Series([3, 4])  # Unnamed series

    # --- Assert ---
    with pytest.raises(ValueError) as excinfo:
        ts_concat([df, s], axis=1)
    assert "must have a 'name' attribute" in str(excinfo.value)


def test_column_sorting() -> None:
    """
    Test that columns maintain their original order when concatenating with axis=1.
    """
    # --- Setup ---
    df1 = pd.DataFrame({"C": [1, 2], "A": [3, 4]})
    df2 = pd.DataFrame({"D": [5, 6], "B": [7, 8]})

    # --- Execute ---
    result = ts_concat([df1, df2], axis=1)

    # --- Assert ---
    # The columns from each DataFrame are kept in their original order
    expected = pd.DataFrame({"C": [1, 2], "A": [3, 4], "D": [5, 6], "B": [7, 8]})
    pd.testing.assert_frame_equal(result, expected)
