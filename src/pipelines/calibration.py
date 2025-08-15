# src/pipelines/calibration.py
import joblib
import sys
from sklearn.base import TransformerMixin, BaseEstimator
from src.exception import CustomException

class RescaleToWaterProperty(BaseEstimator, TransformerMixin):
    def __init__(self, param_path="artifacts/calibration_params.pkl"):
        self.param_path = param_path

    def fit(self, X, y=None):
        try:
            # Load calibration parameters
            self.params_ = joblib.load(self.param_path)
            # Capture feature names to mimic sklearn behavior
            if hasattr(X, "columns"):
                self.feature_names_in_ = X.columns.to_list()
            else:
                # If X is array, require the names be passed somehow
                raise CustomException("RescaleToWaterProperty requires a DataFrame in fit()", sys)
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X):
        try:
            # Work on a copy
            df = X.copy()
            for ch, p in self.params_.items():
                xmin, xmax = p["xmin"], p["xmax"]
                ymin, ymax = p["ymin"], p["ymax"]
                denom = (xmax - xmin) if xmax != xmin else 1.0
                df[ch] = ((df[ch] - xmin) * (ymax - ymin) / denom) + ymin
            return df
        except Exception as e:
            raise CustomException(e, sys)
