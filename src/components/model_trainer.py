import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models, load_object


@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer"""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """Model training component for water sensor fault detection"""

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, preprocessor_path=None):
        """
        Initiate model training process

        Args:
            X_train, y_train, X_test, y_test : Split train/test data
            preprocessor_path : Optional, path to saved preprocessor for logging rescaled features.

        Returns:
            Best model accuracy score on test set
        """
        try:
            logger.info("Initiating Model Training")

            # ===== Optional logging of post-rescale values =====
            if preprocessor_path and os.path.exists(preprocessor_path):
                try:
                    logger.info(f"Loading preprocessor from {preprocessor_path} to log rescaled values")
                    preprocessor = load_object(preprocessor_path)

                    # Expect that your preprocessor pipeline has 'rescale' as the first step
                    if hasattr(preprocessor, "named_steps") and 'rescale' in preprocessor.named_steps:
                        # Create a small DataFrame from first few samples
                        feature_names = [f"Sensor-{i}" for i in range(1, 11)]
                        # We can't directly log from already-scaled X_train (numpy array), so warn if not DF
                        logger.info("Note: Logging expected ranges based on calibration parameters")
                        params = preprocessor.named_steps['rescale'].params_
                        for ch, p in params.items():
                            logger.info(f"{ch} scaled to range {p['ymin']}–{p['ymax']} from wafer range {p['xmin']}–{p['xmax']}")
                    else:
                        logger.warning("Preprocessor has no 'rescale' step — skipping range log")

                except Exception as e:
                    logger.warning(f"Could not log rescaled values: {e}")

            # ===== Define candidate models =====
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "K-Neighbors Classifier": KNeighborsClassifier(),
            }

            # ===== Hyperparameters =====
            params = {
                "Decision Tree": {'criterion': ['gini']},
                "Random Forest": {'n_estimators': [16, 32], 'criterion': ['gini']},
                "Gradient Boosting": {'learning_rate': [0.1], 'n_estimators': [16, 32]},
                "Logistic Regression": {},
                "AdaBoost Classifier": {'learning_rate': [0.1], 'n_estimators': [16, 32]},
                "K-Neighbors Classifier": {'n_neighbors': [5, 7]},
            }

            # ===== Evaluate all models =====
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # ===== Select best model =====
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found (score < 0.6)")

            logger.info(f"Best model: {best_model_name} with score: {best_model_score}")

            # ===== Save model =====
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logger.info(f"Saved best model to {self.model_trainer_config.trained_model_file_path}")

            # ===== Final accuracy =====
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Final model accuracy on test set: {accuracy}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
