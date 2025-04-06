"""Unit tests for the tsutils module."""

import pytest
import pandas as pd
import numpy as np

from tswheel.tsutils import check_stationarity


def check_stationarity_basic_functionality() -> None:
    """
    Test basic functionality of check_stationarity with a simple stationary series.

    Verifies that the function returns a DataFrame with the expected structure.
    """
    # --- Setup ---
    np.random.seed(42)
    # Create a stationary series (white noise)
    data = np.random.normal(0, 1, 100)
    series = pd.Series(data)

    # --- Execute ---
    results = check_stationarity(series)

    # --- Assert ---
    assert isinstance(results, pd.DataFrame)
    expected_columns = [
        "Test",
        "Trend",
        "Test Type",
        "Statistic",
        "P-value",
        "Lags",
        "HAC Lags",
    ]
    assert list(results.columns) == expected_columns

    # Verify that all tests are represented
    expected_tests = [
        "ADF",
        "Phillips-Perron",
        "Zivot-Andrews",
        "KPSS",
        "VarianceRatio",
    ]
    actual_tests = results["Test"].unique().tolist()
    for test in expected_tests:
        assert test in actual_tests


def check_stationarity_with_nonstationary_series() -> None:
    """
    Test check_stationarity with a non-stationary series (random walk).

    Verifies that the function correctly identifies a random walk as non-stationary.
    """
    # --- Setup ---
    np.random.seed(42)
    # Create a non-stationary series (random walk)
    random_walk = np.random.normal(0, 1, 100).cumsum()
    series = pd.Series(random_walk)

    # --- Execute ---
    results = check_stationarity(series)

    # --- Assert ---
    # For ADF, Phillips-Perron, and Zivot-Andrews, high p-values suggest non-stationarity
    adf_results = results[results["Test"] == "ADF"]
    pp_results = results[results["Test"] == "Phillips-Perron"]

    # Check if at least some tests indicate non-stationarity
    # (p-value > 0.05 for ADF, PP, ZA suggests non-stationarity)
    assert any(adf_results["P-value"] > 0.05)
    assert any(pp_results["P-value"] > 0.05)


def check_stationarity_with_stationary_series() -> None:
    """
    Test check_stationarity with a strongly stationary series (AR(1) with |phi| < 1).

    Verifies that the function correctly identifies a stationary AR(1) process.
    """
    # --- Setup ---
    np.random.seed(42)
    # Create a stationary AR(1) process
    n = 200
    phi = 0.5  # AR coefficient (|phi| < 1 ensures stationarity)
    series = np.zeros(n)
    noise = np.random.normal(0, 1, n)

    for t in range(1, n):
        series[t] = phi * series[t - 1] + noise[t]

    # --- Execute ---
    results = check_stationarity(pd.Series(series))

    # --- Assert ---
    # For ADF, Phillips-Perron, and Zivot-Andrews, low p-values suggest stationarity
    adf_results = results[results["Test"] == "ADF"]

    # At least some tests should indicate stationarity
    # (p-value < 0.05 for ADF suggests stationarity)
    assert any(adf_results["P-value"] < 0.05)


def check_stationarity_with_custom_lags() -> None:
    """
    Test check_stationarity with custom lag parameter.

    Verifies that the function correctly uses the specified lag parameter.
    """
    # --- Setup ---
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    series = pd.Series(data)
    custom_lags = 3

    # --- Execute ---
    results = check_stationarity(series, lags=custom_lags)

    # --- Assert ---
    # Check that the ADF test used the specified lags
    adf_results = results[results["Test"] == "ADF"]
    assert all(adf_results["Lags"] == custom_lags)

    # Check that Zivot-Andrews test used the specified lags
    za_results = results[results["Test"] == "Zivot-Andrews"]
    assert all(za_results["Lags"] == custom_lags)


def check_stationarity_with_input_validation() -> None:
    """
    Test that check_stationarity properly validates input.

    Verifies that the function raises TypeError when input is not a pandas Series.
    """
    # --- Setup ---
    invalid_inputs = [
        np.array([1, 2, 3]),  # NumPy array
        [1, 2, 3],  # List
        {"a": 1, "b": 2},  # Dictionary
        42,  # Integer
        "string",  # String
    ]

    # --- Assert ---
    for invalid_input in invalid_inputs:
        with pytest.raises(TypeError):
            check_stationarity(invalid_input)


def check_stationarity_with_seasonal_data() -> None:
    """
    Test check_stationarity with seasonal time series data.

    Verifies the function's behavior with seasonal patterns that may affect stationarity tests.
    """
    # --- Setup ---
    np.random.seed(42)
    n = 100
    t = np.arange(n)

    # Create a seasonal component
    seasonal = 10 * np.sin(2 * np.pi * t / 12)  # Monthly seasonality

    # Add some noise
    noise = np.random.normal(0, 1, n)

    # Combine to create seasonal time series
    data = seasonal + noise
    series = pd.Series(data)

    # --- Execute ---
    results = check_stationarity(series)

    # --- Assert ---
    assert isinstance(results, pd.DataFrame)
    # The seasonal pattern might be identified as non-stationary by some tests
    kpss_results = results[results["Test"] == "KPSS"]
    # KPSS null hypothesis is stationarity, so low p-value suggests non-stationarity
    # Seasonal data might be detected as non-stationary
    assert kpss_results.shape[0] > 0  # Ensure we have KPSS results


def check_stationarity_results_independence() -> None:
    """
    Test that consecutive calls to check_stationarity don't affect each other.

    Verifies that the function's results are independent between calls.
    """
    # --- Setup ---
    np.random.seed(42)
    # Two different series
    stationary_series = pd.Series(np.random.normal(0, 1, 100))
    nonstationary_series = pd.Series(np.random.normal(0, 1, 100).cumsum())

    # --- Execute ---
    results1 = check_stationarity(stationary_series)
    results2 = check_stationarity(nonstationary_series)

    # --- Assert ---
    # Results should be different for different inputs
    assert not results1.equals(results2)

    # Running again with the same input should yield the same results
    results1_repeat = check_stationarity(stationary_series)
    pd.testing.assert_frame_equal(results1, results1_repeat)
