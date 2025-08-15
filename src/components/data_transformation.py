import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object
from src.pipelines.calibration import RescaleToWaterProperty


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    """Data transformation component for water sensor fault detection"""
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> Pipeline:
        """
        Create and return the preprocessing pipeline:
          1) Rescale wafer channels to water-property units
          2) Impute missing values with KNN
          3) Robust scale features
        """
        try:
            logger.info("Data transformation initiated")

            preprocessing_pipeline = Pipeline([
                ('rescale', RescaleToWaterProperty()),     # map into pH, NTU, etc.
                ('imputer', KNNImputer(n_neighbors=3)),     # fill missing values
                ('scaler', RobustScaler())                  # normalize outliers
            ])

            logger.info("Preprocessing pipeline created successfully")
            return preprocessing_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Read train/test CSVs, apply transformations, and save the preprocessor.
        
        Returns:
            train_arr: numpy array of transformed train features + target
            test_arr: numpy array of transformed test features + target
            preprocessor_obj_file_path: path to the saved pipeline object
        """
        try:
            # Read raw data
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logger.info("Read train and test data completed")

            # Build pipeline
            preprocessing_obj = self.get_data_transformer_object()

            # Identify sensor feature columns and target
            target_column_name = "Good/Bad"
            sensor_cols = [f"Sensor-{i}" for i in range(1, 11)]

            # Drop any extra columns if present
            cols_to_drop = [target_column_name] + [c for c in ['Wafers', 'Unnamed: 0'] if c in train_df.columns]

            input_feature_train_df  = train_df.drop(columns=cols_to_drop, errors='ignore')
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df   = test_df.drop(columns=cols_to_drop, errors='ignore')
            target_feature_test_df  = test_df[target_column_name]

            logger.info("Applying preprocessing object on training and testing dataframes")

            # Fit & transform training features, transform test features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df[sensor_cols])
            input_feature_test_arr  = preprocessing_obj.transform(input_feature_test_df[sensor_cols])

            # Combine transformed features with target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr  = np.c_[input_feature_test_arr,  target_feature_test_df.to_numpy()]

            # Save the preprocessing pipeline
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logger.info(f"Saved preprocessing object at {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
