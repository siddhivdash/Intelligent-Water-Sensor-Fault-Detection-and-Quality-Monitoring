import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports
from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """Complete training pipeline for water sensor fault detection"""

    def __init__(self):
        pass

    def start_training(self):
        """
        Start the complete training pipeline

        Returns:
            Model accuracy score
        """
        try:
            logger.info("Training pipeline started")

            # 1. Data Ingestion
            logger.info("Starting Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logger.info("Data Ingestion completed")

            # 2. Data Transformation (will now automatically use RescaleToWaterProperty)
            logger.info("Starting Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logger.info("Data Transformation completed")

            # train_arr and test_arr shape: (n_samples, n_features+1)
            # Last column is target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test   = test_arr[:, :-1], test_arr[:, -1]

            # 3. Model Training
            logger.info("Starting Model Training")
            model_trainer = ModelTrainer()
            accuracy = model_trainer.initiate_model_trainer(
                X_train, y_train,
                X_test, y_test,
                preprocessor_path=preprocessor_path  # optional if trainer needs it
            )
            logger.info("Model Training completed")

            logger.info(f"Training pipeline completed with accuracy: {accuracy}")
            return accuracy

        except Exception as e:
            logger.error("Error in training pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.start_training()
