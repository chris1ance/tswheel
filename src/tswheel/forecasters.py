import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Dict
from sklearn.base import BaseEstimator
from sktime.transformations.series.lag import Lag

pd.set_option("mode.copy_on_write", True)


class DirectForecaster:
    """
    A class implementing direct forecasting using sklearn's LinearRegression.

    This class trains separate models for each forecast horizon h, following the direct
    forecasting approach where each future time step is predicted using its own dedicated model.
    The implementation specifically handles pandas DataFrame and Series inputs, preserving
    index information throughout the forecasting process.
    """

    def __init__(
        self,
        horizons: List[int],
        xlags: List[int] = [0],
        ylags: List[int] | None = None,
        cutoff_date: str | None = None,
    ):
        """
        Initialize the DirectForecaster.

        Args:
            horizons: List of forecast horizons to train models for
        """

        self.ylags = ylags
        self.xlags = xlags
        self.cutoff_date = cutoff_date
        self.horizons = horizons
        self.models: Dict[int, BaseEstimator] = {}
        self.X = None
        self.y = None
        self.X_train: Dict[int, pd.DataFrame] = {}
        self.y_train: Dict[int, pd.DataFrame] = {}
        self.X_test: Dict[int, pd.DataFrame] = {}
        self.y_test: Dict[int, pd.DataFrame] = {}

    def _construct_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Construct training dataset for a specific forecast horizon, optionally adding lags of y.

        Args:
            X: Feature DataFrame with time index
            y: Target Series with matching time index
            horizon: Forecast horizon h
            lags: Number of lags of y to include in the features

        Returns:
            Tuple of (X_h, y_h) where:
                X_h: Features DataFrame for horizon h, including lags of y if specified
                y_h: Target Series for horizon h

        Note:
            This method preserves the index of the input data, which is crucial
            for time series analysis and interpretation of results.
        """
        # Input validation
        if horizon <= 0:
            raise ValueError("Horizon must be positive")

        # Verify that X and y have the same index
        if not X.index.equals(y.index):
            raise ValueError("X and y must have matching indices")

        # Verify index type is datetime or period
        if not (
            isinstance(X.index, (pd.DatetimeIndex, pd.PeriodIndex))
            and isinstance(y.index, (pd.DatetimeIndex, pd.PeriodIndex))
        ):
            raise ValueError("X and y must have datetime or period index")

        # Convert datetime to period if needed
        _X = X.copy()
        _y = y.copy()
        if isinstance(X.index, pd.DatetimeIndex):
            _X.index = _X.index.to_period()
        if isinstance(y.index, pd.DatetimeIndex):
            _y.index = _y.index.to_period()

        if self.ylags is None:
            Z = Lag(self.xlags).fit_transform(_X)
        else:
            Z = pd.concat(
                [
                    Lag(self.ylags).fit_transform(_y.to_frame()),
                    Lag(self.xlags).fit_transform(_X),
                ],
                axis=1,
            )

        Z_h = Z.shift(horizon - 1)
        y_h = _y.rename("y")

        _ = pd.concat([Z_h, y_h], axis=1).dropna()
        X_h = _.drop("y", axis=1)
        y_h = _["y"]

        X_train = X_h[X_h.index <= self.cutoff_date]
        X_test = X_h[X_h.index > self.cutoff_date].iloc[[0]]

        y_train = y_h[y_h.index <= self.cutoff_date]
        y_test = y_h[y_h.index > self.cutoff_date].iloc[[horizon - 1]]

        return X_train, X_test, y_train, y_test

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectForecaster":
        """
        Train separate models for each forecast horizon, optionally including lags of y.

        Args:
            X: Feature DataFrame with time index
            y: Target Series with matching time index
            lags: Number of lags of y to include in the features
            cutoff_date: Date beyond which data should not be used for training

        Returns:
            self: The fitted forecaster

        Note:
            This method stores the feature names from the training data
            to ensure consistent feature ordering in predictions.
        """
        # Validate input types
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")

        # Store full X and y matrices
        self.X = X.copy()
        self.y = y.copy()

        # Train a separate model for each horizon
        for h in self.horizons:
            # Construct dataset for horizon h
            _X_train, _X_test, _y_train, _y_test = self._construct_dataset(X, y, h)

            self.X_train[h] = _X_train
            self.X_test[h] = _X_test
            self.y_train[h] = _y_train
            self.y_test[h] = _y_test

            # Initialize and train model for horizon h
            model = LinearRegression()
            model.fit(_X_train, _y_train)

            # Store the trained model
            self.models[h] = model

        return self

    def predict(self) -> Dict[int, pd.Series]:
        """
        Generate forecasts for all horizons using the trained models.

        Args:
            X: Feature DataFrame with time index. If None, use the full_X matrix stored during fit.

        Returns:
            Dictionary mapping each horizon to its predictions as a pandas Series

        Note:
            The returned predictions preserve the index from the input DataFrame,
            making it easy to align predictions with the original time series.
        """
        if not self.models:
            raise ValueError("Models have not been trained. Call fit() first.")

        predictions = {}
        for h in self.horizons:
            # Get predictions for horizon h and convert to Series with proper index
            pred_values = self.models[h].predict(self.X_test[h])
            predictions[h] = pd.Series(
                pred_values, index=self.y_test[h].index, name=f"h={h}"
            )

        return predictions
