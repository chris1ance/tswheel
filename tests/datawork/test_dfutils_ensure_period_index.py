import pandas as pd
import pytest
from datetime import date

from tswheel.datawork.dfutils import ensure_period_index


# --- Fixtures ---


@pytest.fixture
def df_datetime_daily() -> pd.DataFrame:
    """DataFrame with a daily DatetimeIndex."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame({"data": range(5)}, index=dates)


@pytest.fixture
def df_datetime_monthly() -> pd.DataFrame:
    """DataFrame with a monthly DatetimeIndex."""
    dates = pd.date_range("2023-01-01", periods=3, freq="ME")
    return pd.DataFrame({"data": range(3)}, index=dates)


@pytest.fixture
def df_datetime_irregular() -> pd.DataFrame:
    """DataFrame with an irregular DatetimeIndex (no inferrable freq)."""
    dates = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-05"])
    return pd.DataFrame({"data": range(3)}, index=dates)


@pytest.fixture
def df_period_daily() -> pd.DataFrame:
    """DataFrame with a daily PeriodIndex."""
    periods = pd.period_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame({"data": range(5)}, index=periods)


@pytest.fixture
def df_period_monthly() -> pd.DataFrame:
    """DataFrame with a monthly PeriodIndex."""
    periods = pd.period_range("2023-01", periods=3, freq="M")
    return pd.DataFrame({"data": range(3)}, index=periods)


@pytest.fixture
def df_string_monthly() -> pd.DataFrame:
    """DataFrame with a string index convertible to monthly periods."""
    index = ["2023-01", "2023-02", "2023-03"]
    return pd.DataFrame({"data": range(3)}, index=index)


@pytest.fixture
def df_string_quarterly() -> pd.DataFrame:
    """DataFrame with a string index convertible to quarterly periods."""
    index = ["2023Q1", "2023Q2", "2023Q3"]
    return pd.DataFrame({"data": range(3)}, index=index)


@pytest.fixture
def df_date_objects() -> pd.DataFrame:
    """DataFrame with datetime.date objects as index."""
    index = [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)]
    return pd.DataFrame({"data": range(3)}, index=index)


@pytest.fixture
def df_range_index() -> pd.DataFrame:
    """DataFrame with a RangeIndex."""
    return pd.DataFrame({"data": range(5)})


@pytest.fixture
def df_string_non_parsable() -> pd.DataFrame:
    """DataFrame with a non-parsable string index."""
    index = ["apple", "banana", "cherry"]
    return pd.DataFrame({"data": range(3)}, index=index)


# --- Test Cases ---


def test_already_period_index(df_period_daily: pd.DataFrame):
    """Test that a DataFrame with PeriodIndex is returned unchanged."""
    original_df = df_period_daily.copy()
    result_df = ensure_period_index(original_df)
    assert result_df is original_df  # Should return the same object
    assert isinstance(result_df.index, pd.PeriodIndex)
    assert result_df.index.freqstr == "D"
    pd.testing.assert_frame_equal(result_df, original_df)


# Add a test for existing PeriodIndex with invalid frequency
def test_already_period_index_invalid_freq_raises():
    """Test AssertionError for existing PeriodIndex with disallowed frequency."""
    # Create a PeriodIndex with a frequency not in the allowed list, e.g., '2D'
    periods = pd.period_range("2023-01-01", periods=5, freq="2D")
    df_invalid_freq = pd.DataFrame({"data": range(5)}, index=periods)
    with pytest.raises(
        AssertionError, match="Existing PeriodIndex frequency '2D' is not in"
    ):
        ensure_period_index(df_invalid_freq)


def test_datetime_index_infer_freq(df_datetime_daily: pd.DataFrame):
    """Test conversion from DatetimeIndex with inferrable frequency."""
    original_df = df_datetime_daily.copy()
    result_df = ensure_period_index(original_df)
    assert result_df is not original_df  # Should return a copy
    assert isinstance(result_df.index, pd.PeriodIndex)
    assert result_df.index.freqstr == "D"  # Use freqstr
    # Check data and columns are preserved
    pd.testing.assert_index_equal(result_df.columns, original_df.columns)
    assert (result_df.values == original_df.values).all()
    # Check original is unmodified
    assert isinstance(original_df.index, pd.DatetimeIndex)


def test_datetime_index_specify_freq(df_datetime_monthly: pd.DataFrame):
    """Test conversion from DatetimeIndex using specified frequency."""
    original_df = df_datetime_monthly.copy()
    result_df = ensure_period_index(original_df, freq="M")
    assert result_df is not original_df
    assert isinstance(result_df.index, pd.PeriodIndex)
    assert result_df.index.freqstr == "M"  # Use freqstr
    # Check data and columns are preserved
    pd.testing.assert_index_equal(result_df.columns, original_df.columns)
    assert (result_df.values == original_df.values).all()
    assert isinstance(original_df.index, pd.DatetimeIndex)


def test_datetime_index_override_freq(df_datetime_daily: pd.DataFrame):
    """Test conversion from DatetimeIndex overriding inferred freq with specified freq."""
    original_df = df_datetime_daily.copy()
    # Original is daily, convert to monthly
    result_df = ensure_period_index(original_df, freq="M")
    assert result_df is not original_df
    assert isinstance(result_df.index, pd.PeriodIndex)
    assert result_df.index.freqstr == "M"  # Use freqstr
    # Check data is preserved (index values will change due to freq override)
    assert result_df.shape == original_df.shape
    pd.testing.assert_index_equal(result_df.columns, original_df.columns)
    # Compare values directly, not Series with different indexes
    assert (result_df.values == original_df.values).all()
    assert isinstance(original_df.index, pd.DatetimeIndex)


# Renamed test and updated assertions based on observed pandas behavior
# Updated: This test now expects an AssertionError due to the new assertion in the function
def test_datetime_index_irregular_no_freq_raises_assertion(
    df_datetime_irregular: pd.DataFrame,
):
    """Test that irregular DatetimeIndex raises AssertionError when inferred freq ('2D') is invalid."""
    original_df = df_datetime_irregular.copy()
    # Pandas infers '2D' for this index, which is not in ['B', 'D', 'W', 'M', 'Q', 'Y']
    with pytest.raises(
        AssertionError, match="Resulting PeriodIndex frequency '2D' is not in"
    ):
        ensure_period_index(original_df)
    # Check original is unmodified
    assert isinstance(original_df.index, pd.DatetimeIndex)


def test_datetime_index_irregular_with_freq(df_datetime_irregular: pd.DataFrame):
    """Test conversion of irregular DatetimeIndex when freq is provided."""
    original_df = df_datetime_irregular.copy()
    result_df = ensure_period_index(original_df, freq="D")
    assert result_df is not original_df
    assert isinstance(result_df.index, pd.PeriodIndex)
    assert result_df.index.freqstr == "D"  # Use freqstr
    expected_index = pd.PeriodIndex(
        ["2023-01-01", "2023-01-03", "2023-01-05"], freq="D"
    )
    pd.testing.assert_index_equal(result_df.index, expected_index)
    # Check data and columns are preserved
    pd.testing.assert_index_equal(result_df.columns, original_df.columns)
    assert (result_df.values == original_df.values).all()
    assert isinstance(original_df.index, pd.DatetimeIndex)


def test_string_index_infer_freq(df_string_monthly: pd.DataFrame):
    """Test conversion from convertible string index with inferred frequency."""
    original_df = df_string_monthly.copy()
    result_df = ensure_period_index(original_df)
    assert result_df is not original_df
    assert isinstance(result_df.index, pd.PeriodIndex)
    assert result_df.index.freqstr == "M"  # Use freqstr
    # Check data and columns are preserved
    pd.testing.assert_index_equal(result_df.columns, original_df.columns)
    assert (result_df.values == original_df.values).all()
    assert isinstance(original_df.index, pd.Index)  # Original is object/string index


def test_string_index_specify_freq(df_string_quarterly: pd.DataFrame):
    """Test conversion from convertible string index with specified frequency."""
    original_df = df_string_quarterly.copy()
    result_df = ensure_period_index(original_df, freq="Q")  # Specify Quarter
    assert result_df is not original_df
    assert isinstance(result_df.index, pd.PeriodIndex)
    assert result_df.index.freqstr == "Q-DEC"  # Use freqstr
    # Check data and columns are preserved
    pd.testing.assert_index_equal(result_df.columns, original_df.columns)
    assert (result_df.values == original_df.values).all()
    assert isinstance(original_df.index, pd.Index)


def test_date_object_index_specify_freq(df_date_objects: pd.DataFrame):
    """Test conversion from datetime.date object index with specified frequency."""
    original_df = df_date_objects.copy()
    result_df = ensure_period_index(original_df, freq="M")  # Specify Monthly
    assert result_df is not original_df
    assert isinstance(result_df.index, pd.PeriodIndex)
    assert result_df.index.freqstr == "M"  # Use freqstr
    expected_index = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
    pd.testing.assert_index_equal(result_df.index, expected_index)
    # Check data and columns are preserved
    pd.testing.assert_index_equal(result_df.columns, original_df.columns)
    assert (result_df.values == original_df.values).all()
    assert isinstance(original_df.index, pd.Index)  # Original is object index


def test_date_object_index_infer_freq(df_date_objects: pd.DataFrame):
    """Test that freq can be inferred from date objects via DatetimeIndex."""
    original_df = df_date_objects.copy()
    result_df = ensure_period_index(original_df, freq=None)  # Infer frequency
    assert result_df is not original_df
    assert isinstance(result_df.index, pd.PeriodIndex)
    # pd.to_datetime infers 'MS' from these dates, which converts to 'M' PeriodIndex
    assert result_df.index.freqstr == "M"  # Corrected expected freq
    expected_index = pd.PeriodIndex(
        ["2023-01", "2023-02", "2023-03"], freq="M"
    )  # Corrected expected index
    pd.testing.assert_index_equal(result_df.index, expected_index)
    # Check data and columns are preserved
    pd.testing.assert_index_equal(result_df.columns, original_df.columns)
    assert (result_df.values == original_df.values).all()
    assert isinstance(original_df.index, pd.Index)  # Original is object index


def test_non_convertible_index_raises(df_range_index: pd.DataFrame):
    """Test TypeError for non-convertible index types like RangeIndex."""
    # Expects TypeError because the function checks dtype before pd.to_datetime
    with pytest.raises(TypeError, match="Index type RangeIndex.*cannot be converted"):
        ensure_period_index(df_range_index)


def test_non_parsable_string_index_raises(df_string_non_parsable: pd.DataFrame):
    """Test TypeError for non-parsable string index."""
    # This should still raise TypeError, but now from the pd.to_datetime call failing
    with pytest.raises(
        TypeError, match="Could not parse index values as datetime objects"
    ):
        ensure_period_index(df_string_non_parsable, freq="D")  # Freq doesn't help here


def test_invalid_input_type_raises():
    """Test TypeError if input is not a DataFrame."""
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
        ensure_period_index("not a dataframe")  # type: ignore


def test_copy_on_write_behavior(df_datetime_daily: pd.DataFrame):
    """Verify that the original DataFrame is not modified due to CoW or explicit copy."""
    original_df = df_datetime_daily.copy()
    original_index_id = id(original_df.index)

    result_df = ensure_period_index(original_df, freq="D")

    # Check result is different object with different index object
    assert result_df is not original_df
    assert id(result_df.index) != original_index_id

    # Check original DataFrame remains unchanged
    assert isinstance(original_df.index, pd.DatetimeIndex)
    assert id(original_df.index) == original_index_id
