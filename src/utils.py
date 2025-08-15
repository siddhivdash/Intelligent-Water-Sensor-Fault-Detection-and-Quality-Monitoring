# here we create the pkl files

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import dill
#import yaml
#from box import ConfigBox
from pathlib import Path
from typing import Any

from src.logger import logger
from src.exception import CustomException


#def read_yaml(path_to_yaml: Path) -> ConfigBox:
#    """reads yaml file and returns

#    Args:
 #       path_to_yaml (str): path like input

    #Raises:
       # ValueError: if yaml file is empty
       # e: empty file

   # Returns:
      #  ConfigBox: ConfigBox type
   # """
    #try:  
       # with open(path_to_yaml) as yaml_file:
        #    content = yaml.safe_load(yaml_file)
       # if content is None:
       #     raise ValueError("yaml file is empty")
        #logger.info(f"yaml file: {path_to_yaml} loaded successfully")
       # return ConfigBox(content)
   # except Exception as e:
    #    raise e



def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def save_object(file_path, obj):
    """
    Save object to a pickle file
    
    Args:
        file_path: Path where object will be saved
        obj: Object to be saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logger.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load object from pickle file
    
    Args:
        file_path: Path of the pickle file
        
    Returns:
        Loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    Args:
        file_path: str location of file to save
        array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    Args:
        file_path: str location of file to load
    Returns:
        np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models and return their scores
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        models: Dictionary of models to evaluate
        param: Parameters for hyperparameter tuning
        
    Returns:
        Dictionary with model scores
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # GridSearchCV for hyperparameter tuning
            from sklearn.model_selection import GridSearchCV
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Model prediction
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"