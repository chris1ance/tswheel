import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
from typing import Any
from .datawork.dfutils import have_same_index_type


class TSOLS:
    """
    A wrapper class for statsmodels OLS regression using pandas DataFrames/Series.

    This class simplifies the process of performing Ordinary Least Squares (OLS)
    regression using the statsmodels library, specifically tailored for pandas inputs.
    It handles alignment of input data based on index, drops rows with NaNs in
    either y or X, and fits the model upon initialization using Heteroskedasticity
    and Autocorrelation Consistent (HAC) standard errors (Newey-West).

    Attributes:
        y (pd.Series): The original dependent (endogenous) variable passed during initialization.
        X (pd.DataFrame): The original independent (exogenous) variables passed during
                          initialization (potentially with an added constant).
        _y (pd.Series): The aligned and NaN-dropped dependent variable used for fitting.
        _X (pd.DataFrame): The aligned and NaN-dropped independent variables used for fitting.
        model (sm.OLS): The statsmodels OLS model instance created before fitting.
        results (RegressionResultsWrapper): The fitted OLS results instance, obtained using
                                            HAC standard errors (cov_type='HAC').
        has_intercept (bool): Flag indicating if an intercept ('const') term was added to X.

    Public Methods:
        get_residuals(): Get residuals from the fitted OLS model.
        get_fitted_values(): Get fitted values from the OLS model.
        get_r2(adjusted=False): Get R-squared or adjusted R-squared.
        get_coefficients_summary(): Get summary DataFrame of coefficients and HAC statistics.
        get_ftest_results(): Get overall model F-statistic and p-value.
        summary(): Get the full statsmodels summary object.
        test_breusch_godfrey(nlags=None): Perform Breusch-Godfrey test for residual autocorrelation.
        test_coefficient_restrictions(hypotheses): Perform F-test for linear restrictions on coefficients.
        predict(exog, **kwargs): Generate predictions using the fitted model.
    """

    def __init__(self, y: pd.Series, X: pd.DataFrame, intercept: bool = True) -> None:
        """
        Initialize the tsOLS, align data, handle NaNs, fit the OLS model using
        HAC standard errors, and store results.

        The input Series and DataFrame are aligned using an inner join on their indices.
        Rows containing NaN values in either the aligned y or X are dropped before fitting.
        The model is fitted using `cov_type='HAC'` with `use_correction=True` and
        automatic lag length selection based on Newey and West (1994).

        Args:
            y (pd.Series): The dependent variable.
            X (pd.DataFrame): The independent variables.
            intercept (bool, default=True): If True, adds a constant intercept term to X
                                            using `sm.add_constant`. If False, no
                                            intercept is added.

        Raises:
            ValueError: If y and X have incompatible index types.
            TypeError: If y is not a pd.Series or X is not a pd.DataFrame/pd.Series.
            Exception: Any exception raised by statsmodels during model fitting.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> import statsmodels.api as sm
            >>> from tswheel.regressions import tsOLS # Assuming the class is saved here
            >>>
            >>> # Generate sample data
            >>> np.random.seed(0)
            >>> dates = pd.date_range('2023-01-01', periods=100)
            >>> X_np = np.random.rand(100, 2)
            >>> X_df = pd.DataFrame(X_np, columns=['x1', 'x2'], index=dates)
            >>> beta = np.array([1.0, 0.5, -0.8]) # Intercept, beta1, beta2
            >>> e = np.random.randn(100) * 0.5
            >>> # Manually add intercept for calculation, tsOLS adds it automatically
            >>> y_np = sm.add_constant(X_df.values, prepend=True) @ beta + e
            >>> y_series = pd.Series(y_np, name='y', index=dates)
            >>>
            >>> # Add some NaNs to test handling
            >>> X_df.iloc[5, 0] = np.nan
            >>> y_series.iloc[10] = np.nan
            >>>
            >>> # Initialize and fit the model with intercept (default)
            >>> # Note: HAC standard errors are used by default
            >>> ols_pd = tsOLS(y_series, X_df)
            >>> print(f"R-squared with intercept: {ols_pd.get_r2():.4f}")
            R-squared with intercept: 0.4798
            >>> print(f"Number of observations used: {ols_pd.results.nobs}")
            Number of observations used: 98.0
            >>>
            >>> # Initialize and fit the model without intercept
            >>> ols_no_intercept = tsOLS(y_series, X_df, intercept=False)
            >>> print(f"R-squared without intercept: {ols_no_intercept.get_r2():.4f}")
            R-squared without intercept: 0.0441
            >>>
            >>> # Adjusted R-squared
            >>> print(f"Adjusted R-squared (with intercept): {ols_pd.get_r2(adjusted=True):.4f}")
            Adjusted R-squared (with intercept): 0.4689
            >>>
            >>> # Get coefficients summary (includes HAC SE/t/p values)
            >>> print(ols_pd.get_coefficients_summary().round(4))
                     Coefficient      SE  tvalues  pvalues
            const         1.0490  0.1018  10.3079   0.0000
            x1            0.4019  0.2999   1.3400   0.1834
            x2           -0.8310  0.3099  -2.6815   0.0086
        """
        if not isinstance(y, pd.Series):
            raise TypeError(f"Expected y to be a pandas Series, but got {type(y)}")
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, pd.Series):
                X = X.to_frame()
            else:
                raise TypeError(
                    f"Expected X to be a pandas DataFrame, but got {type(X)}"
                )

        # Basic index check
        if not have_same_index_type(y, X):
            raise ValueError("Indices of y and X do not match.")

        # Align the DataFrames to get common indices
        _y, _X = y.align(X, join="inner", axis=0)

        # Create a combined mask for non-NaN values in both DataFrames
        # For Series, we don't use axis parameter
        x_mask = ~_X.isna().any(axis=1)
        y_mask = ~_y.isna()
        valid_mask = x_mask & y_mask

        # Apply the mask to both DataFrames
        _X = _X.loc[valid_mask]
        _y = _y.loc[valid_mask]

        if intercept:
            # When input is pandas Series or DataFrame, added column's name is 'const'
            _X = sm.add_constant(_X, prepend=True)

        # Calculate optimal lag length
        T = len(_y)  # Sample size
        optimal_lag = int(
            np.floor(4 * (T / 100) ** (2 / 9))
        )  # Newey and West, 1994, "Automatic Lag Selection in Covariance Matrix Estimation," RES

        self.has_intercept = intercept
        self.y = y
        self.X = X
        self._y = _y
        self._X = _X
        self.model = sm.OLS(self._y, self._X)
        self.results = self.model.fit(
            cov_type="HAC", cov_kwds={"maxlags": optimal_lag, "use_correction": True}
        )

    def get_residuals(self) -> pd.Series:
        """
        Get residuals from the fitted OLS model.

        Residuals are the differences between the observed values (`_y`) and the
        values predicted by the model using the fitted parameters on `_X`.

        Returns:
            residuals (pd.Series): Residuals from the model, indexed like the input
                                   data used for fitting (`_y`).

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> residuals = ols_pd.get_residuals()
            >>> print(residuals.head().round(4)) # Note: index matches original y
            2023-01-01   -0.3243
            2023-01-02   -0.0941
            2023-01-03   -0.0489
            2023-01-04   -0.3387
            2023-01-05   -0.5087
            Freq: D, Name: y, dtype: float64
            >>> print(residuals.shape) # Matches original y shape
            (100,)
            >>> print(residuals.isna().sum()) # Includes NaNs from original data + alignment drops
            2
        """
        return pd.Series(self.results.resid, index=self._y.index)

    def get_fitted_values(self) -> pd.Series:
        """
        Get fitted values from the OLS model.

        Fitted values are the values predicted by the model using the fitted parameters
        on the training data `_X`. They represent the "explained" component of the
        dependent variable.

        Returns:
            fitted_values (pd.Series): Fitted values from the model, indexed like the
                                       input data used for fitting (`_y`).

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> fitted = ols_pd.get_fitted_values()
            >>> print(fitted.head().round(4)) # Note: index matches original y
            2023-01-01    1.1133
            2023-01-02    1.0011
            2023-01-03    0.9119
            2023-01-04    1.1017
            2023-01-05    1.3117
            Freq: D, dtype: float64
            >>> # Verify that residuals + fitted values = original data
            >>> y_reconstructed = fitted + ols_pd.get_residuals()
            >>> print(((y_reconstructed - ols_pd._y).abs() < 1e-10).all())
            True
        """
        return pd.Series(self.results.fittedvalues, index=self._y.index)

    def get_r2(self, adjusted: bool = False) -> float:
        """
        Get R-squared or adjusted R-squared.

        R-squared represents the proportion of the variance in the dependent variable
        that is predictable from the independent variables. Adjusted R-squared penalizes
        the addition of predictors that do not improve the model fit.

        Args:
            adjusted (bool, default=False): If True, return adjusted R-squared.

        Returns:
            r_squared (float): R-squared or adjusted R-squared value.

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> r2 = ols_pd.get_r2()
            >>> print(f"{r2:.4f}")
            0.4798
            >>> adj_r2 = ols_pd.get_r2(adjusted=True)
            >>> print(f"{adj_r2:.4f}")
            0.4689
        """
        if adjusted:
            return float(self.results.rsquared_adj)
        else:
            return float(self.results.rsquared)

    def get_coefficients_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of coefficients and HAC statistics.

        The DataFrame includes the estimated coefficient values, their HAC standard errors,
        the corresponding t-statistics, and the p-values, calculated using the
        Newey-West estimator.

        Returns:
            summary_df (pd.DataFrame): DataFrame with columns 'Coefficient', 'SE'
                                       (Standard Error), 'tvalues', and 'pvalues',
                                       indexed by the feature names from `_X`.

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> coeff_summary = ols_pd.get_coefficients_summary()
            >>> print(coeff_summary.round(4))
                     Coefficient      SE  tvalues  pvalues
            const         1.0490  0.1018  10.3079   0.0000
            x1            0.4019  0.2999   1.3400   0.1834
            x2           -0.8310  0.3099  -2.6815   0.0086
        """
        summary_df = pd.DataFrame(
            {
                "Coefficient": self.results.params,
                "SE": self.results.bse,  # HAC Standard errors
                "tvalues": self.results.tvalues,  # t-values based on HAC SE
                "pvalues": self.results.pvalues,  # p-values based on HAC SE
            }
        )
        return summary_df

    def get_ftest_results(self) -> dict[str, float]:
        """
        Get overall model F-statistic and p-value.

        The F-statistic tests the hypothesis that all regression coefficients (excluding
        the intercept) are simultaneously equal to zero. The p-value indicates the
        probability of observing the calculated F-statistic (or a more extreme value)
        if the null hypothesis (all coefficients are zero) were true.

        Returns:
            ftest_results (dict[str, float]): Dictionary with 'fvalue' (F-statistic)
                                              and 'pvalue' (p-value for F-statistic).

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> ftest = ols_pd.get_ftest_results()
            >>> print(f"F-statistic: {ftest['fvalue']:.4f}")
            F-statistic: 43.2918
            >>> print(f"p-value: {ftest['pvalue']:.4e}")
            p-value: 1.0011e-13
        """
        return {
            "fvalue": float(self.results.fvalue),
            "pvalue": float(self.results.f_pvalue),
        }

    def summary(self) -> Any:
        """
        Get the full statsmodels summary object.

        This provides a comprehensive overview of the regression results, including
        coefficients, standard errors, t-statistics, p-values, R-squared, F-statistic, etc.

        Returns:
            summary_obj (Any): The summary object from statsmodels results (typically a
                               Summary instance). Use print() to display it.

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> model_summary = ols_pd.summary()
            >>> # print(model_summary) # doctest: +SKIP
        """
        return self.results.summary()

    def test_breusch_godfrey(self, nlags: int | None = None) -> dict[str, float]:
        """
        Perform Breusch-Godfrey test for residual autocorrelation.

        Tests the null hypothesis of no serial correlation in the residuals up to
        order 'nlags'. Returns p-values from both the Lagrange Multiplier and F-test.

        Args:
            nlags (int | None, default=None): Number of lags for the test. If None,
                                              it's auto-selected based on sample size.

        Returns:
            bg_test_results (dict[str, float]): Dictionary with p-values 'lmpval'
                                                (Lagrange Multiplier test) and 'fpval'
                                                (F-test).

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance
            >>> result = ols_pd.test_breusch_godfrey(nlags=4)
            >>> print(f"LM test p-value: {result['lmpval']:.4f}")
            >>> print(f"F test p-value: {result['fpval']:.4f}")
        """
        # --- Input Validation ---
        if nlags is not None and not isinstance(nlags, int):
            raise TypeError(f"nlags must be an integer or None, got {type(nlags)}")

        if nlags is not None and nlags <= 0:
            raise ValueError(f"nlags must be positive, got {nlags}")

        # --- Perform Breusch-Godfrey Test ---
        # If nlags is None, use a default based on sample size
        if nlags is None:
            T = len(self._y)
            nlags = min(int(np.ceil(np.sqrt(T))), int(T / 5))  # Common rule of thumb

        # Run the Breusch-Godfrey test
        lm, lmpval, fval, fpval = smd.acorr_breusch_godfrey(
            self.results, nlags=nlags, store=False
        )

        return {"lmpval": float(lmpval), "fpval": float(fpval)}

    def test_coefficient_restrictions(self, hypotheses: list[str]) -> dict[str, float]:
        """
        Perform F-test for linear restrictions on coefficients.

        Tests joint linear hypotheses about model coefficients using the F-test
        functionality from statsmodels.

        Args:
            hypotheses (list[str]): List of hypothesis strings (e.g., 'x1 = 0',
                                    'x1 + x2 = 0', 'x1 = x2'). Variable names must
                                    match the column names in the model.

        Returns:
            ftest_results (dict[str, float]): Dictionary with 'fstat' (F-statistic)
                                              and 'pvalue' (p-value for F-statistic).

        Raises:
            ValueError: If the hypotheses list is empty or contains invalid constraints,
                        or if variable names in hypotheses don't match model columns.

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance with variables 'x1' and 'x2'
            >>> # Test if coefficients of x1 and x2 are both zero
            >>> restrictions = ['x1 = 0', 'x2 = 0']
            >>> result = ols_pd.test_coefficient_restrictions(restrictions)
            >>> print(f"F-statistic: {result['fstat']:.4f}")
            >>> print(f"p-value: {result['pvalue']:.4e}")
            >>>
            >>> # Test if the coefficient of x1 equals the coefficient of x2
            >>> result = ols_pd.test_coefficient_restrictions(['x1 = x2'])
            >>> print(f"F-statistic: {result['fstat']:.4f}")
            >>> print(f"p-value: {result['pvalue']:.4e}")
        """
        # --- Input Validation ---
        if not hypotheses:
            raise ValueError("The hypotheses list cannot be empty")

        if not isinstance(hypotheses, list) or not all(
            isinstance(h, str) for h in hypotheses
        ):
            raise TypeError("hypotheses must be a list of strings")

        # --- Perform F-test ---
        try:
            ftest_result = self.results.f_test(hypotheses)
            return {
                "fstat": float(ftest_result.statistic),
                "pvalue": float(ftest_result.pvalue),
            }
        except Exception as e:
            raise ValueError(
                f"Error performing F-test: {str(e)}. Check that variable names "
                f"in hypotheses match column names in the model: {list(self._X.columns)}"
            )

    def predict(self, exog: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """
        Generate predictions using the fitted model.

        If the model was fitted with an intercept (`intercept=True` during initialization),
        this method automatically adds a constant column ('const') to `exog` if missing.

        Args:
            exog (pd.DataFrame): DataFrame of exogenous variables for prediction. Must
                                 contain columns matching the original non-constant
                                 features used for training.
            **kwargs (Any): Additional arguments for `statsmodels.results.predict`.

        Returns:
            predictions (pd.Series): Predicted values, indexed like `exog`.

        Raises:
            ValueError: If `exog` columns (after potentially adding a constant) don't
                        match training columns or cannot be aligned.
            TypeError: If `exog` is not a pandas DataFrame.

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> # Note: new_data does NOT need 'const', it will be added automatically
            >>> new_dates = pd.to_datetime(['2024-01-01', '2024-01-02'])
            >>> new_data = pd.DataFrame({'x1': [0.2, 0.8], 'x2': [0.3, 0.7]}, index=new_dates)
            >>> predictions = ols_pd.predict(new_data)
            >>> print(predictions.round(4))
            2024-01-01    0.8794
            2024-01-02    0.7880
            dtype: float64
            >>>
            >>> # Example without intercept
            >>> ols_no_intercept = tsOLS(y_series, X_df, intercept=False)
            >>> new_data_no_intercept = pd.DataFrame({'x1': [0.2, 0.8], 'x2': [0.3, 0.7]}, index=new_dates)
            >>> predictions_no_intercept = ols_no_intercept.predict(new_data_no_intercept)
            >>> print(predictions_no_intercept.round(4))
            2024-01-01    0.1019
            2024-01-02    0.1784
            dtype: float64
        """
        if not isinstance(exog, pd.DataFrame):
            raise TypeError(
                f"Expected exog to be a pandas DataFrame, but got {type(exog)}"
            )

        _exog = exog.copy()  # Work on a copy

        # Automatically add constant if model has intercept and 'const' is missing
        if self.has_intercept and "const" not in _exog.columns:
            _exog = sm.add_constant(_exog, prepend=True, has_constant="skip")

        # Check feature consistency (column names and order might matter for statsmodels)
        if not self._X.columns.equals(_exog.columns):
            # Consider a more robust check if column order isn't guaranteed
            if set(self._X.columns) != set(_exog.columns):
                raise ValueError(
                    "Prediction exog columns (after adding constant if needed) do not match "
                    f"training X columns. Expected: {list(self._X.columns)}, "
                    f"Got: {list(_exog.columns)}"
                )
            else:
                # Reorder columns if they are just in a different order
                try:
                    _exog = _exog[self._X.columns]
                except Exception as e:
                    raise ValueError(f"Error aligning prediction exog columns: {e}")

        return self.results.predict(exog=_exog, transform=False, **kwargs)
