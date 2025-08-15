import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logger

class CustomData:
    """Custom data class for handling input data."""
    
    def __init__(
        self,
        sensor_1: float,
        sensor_2: float,
        sensor_3: float,
        sensor_4: float,
        sensor_5: float,
        sensor_6: float,
        sensor_7: float,
        sensor_8: float,
        sensor_9: float,
        sensor_10: float
    ):
        self.sensor_1 = sensor_1
        self.sensor_2 = sensor_2
        self.sensor_3 = sensor_3
        self.sensor_4 = sensor_4
        self.sensor_5 = sensor_5
        self.sensor_6 = sensor_6
        self.sensor_7 = sensor_7
        self.sensor_8 = sensor_8
        self.sensor_9 = sensor_9
        self.sensor_10 = sensor_10

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Convert custom data to a pandas DataFrame
        with columns 'Sensor-1' ... 'Sensor-10'.
        """
        try:
            data_dict = {
                f"Sensor-{i}": [getattr(self, f"sensor_{i}")]
                for i in range(1, 11)
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    """Prediction pipeline for water sensor fault detection."""
    
    def __init__(self):
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        self.model_path       = "artifacts/model.pkl"

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Load preprocessor & model, pad missing features, transform, and predict.
        
        Args:
            input_df: DataFrame containing only 'Sensor-1'...'Sensor-10'.
            
        Returns:
            numpy array of predictions.
        """
        try:
            # 1. Load preprocessor and model objects
            logger.info("Loading preprocessor and model")
            preprocessor = load_object(file_path=self.preprocessor_path)
            model        = load_object(file_path=self.model_path)

            # 2. Determine all features seen during training
            expected_features = list(preprocessor.feature_names_in_)

            # 3. Pad any missing columns with NaN
            for feat in expected_features:
                if feat not in input_df.columns:
                    input_df[feat] = np.nan

            # 4. Reorder columns to match training order
            input_df = input_df[expected_features]

            logger.info("Applying preprocessing to input data")
            data_transformed = preprocessor.transform(input_df)

            logger.info("Performing prediction")
            preds = model.predict(data_transformed)

            return preds

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            raise CustomException(e, sys)
