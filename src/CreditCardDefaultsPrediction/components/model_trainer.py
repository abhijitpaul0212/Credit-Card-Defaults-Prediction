# model_trainer.py

import os
import sys
import numpy as np
from dataclasses import dataclass
from src.CreditCardDefaultsPrediction.logger import logging
from src.CreditCardDefaultsPrediction.exception import CustomException
from src.CreditCardDefaultsPrediction.utils.utils import Utils

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    """
    This is configuration class for Model Trainer
    """
    trained_model_obj_path: str = os.path.join("artifacts", "model.pkl")
    trained_model_report_path: str = os.path.join('artifacts', 'model_report.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = Utils()

    def initiate_model_training(self, train_dataframe, test_dataframe):
        try:
            logging.info("Splitting Dependent and Independent features from train and validation & test dataset")

            X_train, y_train, X_test, y_test = (
                train_dataframe.iloc[:, :-1],
                train_dataframe.iloc[:, -1],
                test_dataframe.iloc[:, :-1],
                test_dataframe.iloc[:, -1])
            
            models = {
                    'DecisionTree': DecisionTreeClassifier(),
                    'SVM': SVC(),
                    'LogisticRegression': LogisticRegression(),
                    'NearestNeighbors': KNeighborsClassifier(),
                    'GradientBoosting': GradientBoostingClassifier(),
                    'AdaBoost': AdaBoostClassifier(),
                    'NaiveBayes': GaussianNB()
                }
            
            best_model = self.utils.evaluate_models(models, X_train, y_train, X_test, y_test, metric="accuracy")
            
            self.utils.save_object(
                 file_path=self.model_trainer_config.trained_model_obj_path,
                 obj=best_model
            )       

        except Exception as e:
            raise CustomException(e, sys)
