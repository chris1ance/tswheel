import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Dict
from sklearn.base import BaseEstimator


class DirectForecaster:
    """
    A class implementing direct forecasting using sklearn's LinearRegression.

    This class trains separate models for each forecast horizon h, following the direct
    forecasting approach where each future time step is predicted using its own dedicated model.
    The implementation specifically handles pandas DataFrame and Series inputs, preserving
    index information throughout the forecasting process.
    """

    def __init__(self, horizons: List[int]):
        """
        Initialize the DirectForecaster.

        Args:
            horizons: List of forecast horizons to train models for
        """
        self.horizons = horizons
        self.models: Dict[int, BaseEstimator] = {}
        self.feature_names = None  # Store feature names from training DataFrame

    def _construct_dataset(
        self, X: pd.DataFrame, y: pd.Series, horizon: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Construct training dataset for a specific forecast horizon.

        Args:
            X: Feature DataFrame with time index
            y: Target Series with matching time index
            horizon: Forecast horizon h

        Returns:
            Tuple of (X_h, y_h) where:
                X_h: Features DataFrame for horizon h
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
        if isinstance(X.index, pd.DatetimeIndex):
            _X = X.copy()
            X.index = X.index.to_period()
        if isinstance(y.index, pd.DatetimeIndex):
            _y = y.copy()
            y.index = y.index.to_period()

        # Create dataset D_h = {(x_t, y_{t+h})}
        # We use index alignment to ensure proper time matching
        X_h = X.shift(horizon - 1)
        y_h = y.rename("y")

        data = pd.concat([X_h, y_h], axis=1).dropna()
        X_h = data.drop("y", axis=1)
        y_h = data["y"]

        # Verify that the resulting datasets align properly
        if not X_h.index.equals(y_h.index):
            raise ValueError(f"Index mismatch after shifting for horizon {horizon}")

        return X_h, y_h

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectForecaster":
        """
        Train separate models for each forecast horizon.

        Args:
            X: Feature DataFrame with time index
            y: Target Series with matching time index

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

        # Store feature names for later use
        self.feature_names = X.columns.tolist()

        # Train a separate model for each horizon
        for h in self.horizons:
            # Construct dataset for horizon h
            X_h, y_h = self._construct_dataset(X, y, h)

            # Initialize and train model for horizon h
            model = LinearRegression()
            model.fit(X_h, y_h)

            # Store the trained model
            self.models[h] = model

        return self

    def predict(self, X: pd.DataFrame) -> Dict[int, pd.Series]:
        """
        Generate forecasts for all horizons using the trained models.

        Args:
            X: Feature DataFrame with time index

        Returns:
            Dictionary mapping each horizon to its predictions as a pandas Series

        Note:
            The returned predictions preserve the index from the input DataFrame,
            making it easy to align predictions with the original time series.
        """
        if not self.models:
            raise ValueError("Models have not been trained. Call fit() first.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Verify that X has the expected features
        if not set(self.feature_names).issubset(set(X.columns)):
            raise ValueError("Input DataFrame is missing features used during training")

        # Reorder columns to match training data
        X = X[self.feature_names]

        predictions = {}
        for h in self.horizons:
            # Get predictions for horizon h and convert to Series with proper index
            pred_values = self.models[h].predict(X)
            predictions[h] = pd.Series(pred_values, index=X.index, name=f"h={h}")

        return predictions
