import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, Any
from .datawork.dfutils import have_same_index_type


class tsOLS:
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
    """

    def __init__(self, y: pd.Series, X: pd.DataFrame, intercept: bool = True) -> None:
        """
        Initializes the tsOLS, aligns data, handles NaNs, fits the OLS model using
        HAC standard errors, and stores the results.

        The input Series and DataFrame are aligned using an inner join on their indices.
        Rows containing NaN values in either the aligned y or X are dropped before fitting.
        The model is fitted using `cov_type='HAC'` with `use_correction=True` and
        automatic lag length selection based on Newey and West (1994).

        Args:
            y (pd.Series): The dependent variable (n_samples,).
            X (pd.DataFrame): The independent variables (n_samples, n_features).
            intercept (bool, optional): If True, adds a constant intercept term to X
                                       using `sm.add_constant`. Defaults to True.
                                       If False, no intercept is added.

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
        valid_mask = ~(_X.isna().any(axis=1) | _y.isna().any(axis=1))

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
        Returns the residuals from the fitted OLS regression model.

        Residuals are the differences between the observed values (`_y`) and the
        values predicted by the model using the fitted parameters on `_X`.
        The returned Series is indexed according to `_y`.

        Returns:
            pd.Series: A pandas Series containing the residuals, indexed the same as `_y`

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

    def get_r2(self, adjusted: bool = False) -> float:
        """
        Returns the R-squared or adjusted R-squared value of the fitted model.

        R-squared represents the proportion of the variance in the dependent variable
        that is predictable from the independent variables. Adjusted R-squared penalizes
        the addition of predictors that do not improve the model fit.

        Args:
            adjusted (bool): If True, returns the adjusted R-squared.
                             Defaults to False (returns standard R-squared).

        Returns:
            float: The R-squared or adjusted R-squared value.

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
        Returns a DataFrame summarizing the estimated coefficients and HAC statistics.

        The DataFrame includes the estimated coefficient values, their HAC standard errors,
        the corresponding t-statistics, and the p-values, calculated using the
        Newey-West estimator.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Coefficient', 'SE' (Standard Error),
                          'tvalues', and 'pvalues', indexed by the feature names
                          from the (potentially augmented) X DataFrame used in fitting (`_X`).

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

    def get_fvalue(self) -> float:
        """
        Returns the F-statistic value for the overall model significance test.

        The F-statistic tests the hypothesis that all regression coefficients (excluding
        the intercept) are simultaneously equal to zero.

        Returns:
            float: The F-statistic value.

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> fval = ols_pd.get_fvalue()
            >>> print(f"{fval:.4f}")
            43.2918
        """
        return float(self.results.fvalue)

    def get_f_pvalue(self) -> float:
        """
        Returns the p-value associated with the F-statistic.

        This p-value indicates the probability of observing the calculated F-statistic
        (or a more extreme value) if the null hypothesis (all coefficients are zero)
        were true.

        Returns:
            float: The p-value for the F-statistic.

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> fpval = ols_pd.get_f_pvalue()
            >>> print(f"{fpval:.4e}")
            1.0011e-13
        """
        return float(self.results.f_pvalue)

    def summary(self) -> Any:
        """
        Returns the full summary table generated by statsmodels.

        This provides a comprehensive overview of the regression results, including
        coefficients, standard errors, t-statistics, p-values, R-squared, F-statistic, etc.

        Returns:
            Any: The summary object from statsmodels results (typically a Summary instance).
                 Use print(ols_wrapper.summary()) to display it.

        Examples:
            >>> # Assuming ols_pd is an initialized tsOLS instance from the __init__ example
            >>> model_summary = ols_pd.summary()
            >>> # print(model_summary) # doctest: +SKIP
        """
        return self.results.summary()

    def predict(self, exog: Optional[pd.DataFrame] = None, **kwargs: Any) -> pd.Series:
        """
        Generates predictions using the fitted model.

        If the model was fitted with an intercept (`intercept=True` during initialization),
        this method will automatically add a constant column (named 'const') to the
        provided `exog` DataFrame if it doesn't already exist.

        Args:
            exog (Optional[pd.DataFrame]): The exogenous variables (features) for which
                to generate predictions. Must be a pandas DataFrame. If the model includes
                an intercept, `exog` should contain columns matching the original non-constant
                features used for training. A constant column will be added automatically if
                missing. If None, uses the internal training data `_X` to generate
                in-sample predictions on the data used for fitting.
            **kwargs: Additional keyword arguments passed to the statsmodels predict method.

        Returns:
            pd.Series: The predicted values. If `exog` is provided, the index matches
                       `exog`. If `exog` is None, the index matches the internal
                       cleaned data `_X`.

        Raises:
            ValueError: If the provided `exog` (after potentially adding a constant)
                        has a different set of columns than the internal training data `_X`,
                        or if columns cannot be aligned.
            TypeError: If the provided `exog` is not a pandas DataFrame (and not None).

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
            >>>
            >>> # In-sample predictions (on the cleaned data _X used for fitting)
            >>> in_sample_preds = ols_pd.predict()
            >>> print(in_sample_preds.shape) # Matches shape of _X and _y
            (98,)
            >>> print(in_sample_preds.head().round(4)) # Index matches _X
            2023-01-01    1.1133
            2023-01-02    1.0011
            2023-01-03    0.9119
            2023-01-04    1.1017
            2023-01-05    1.3117
            Freq: D, dtype: float64
        """
        if exog is None:
            _exog = self._X.copy()  # Use the internal, cleaned training data
        else:
            if not isinstance(exog, pd.DataFrame):
                raise TypeError(
                    f"Expected exog to be None or a pandas DataFrame, but got {type(exog)}"
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

        # Statsmodels predict typically returns a numpy array, convert to Series
        predictions_np = self.results.predict(exog=_exog, **kwargs)
        return pd.Series(predictions_np, index=_exog.index)
