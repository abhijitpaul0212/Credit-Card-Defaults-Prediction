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


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = Utils()

    def initiate_model_training(self, train_dataframe, val_dataframe):
        try:
            logging.info("Splitting Dependent and Independent features from train and validation & test dataset")

            X_train, y_train, X_val, y_val = (
                train_dataframe.iloc[:, :-1],
                train_dataframe.iloc[:, -1],
                val_dataframe.iloc[:, :-1],
                val_dataframe.iloc[:, -1])

            models = (
                {
                    GaussianNB(): {
                        'var_smoothing': np.logspace(0,-9, num=100)
                        }},
                # {
                #     LogisticRegression(): {
                #         'penalty': ['l2'],
                #         'C': [0.01, 0.1, 1, 10, 100]
                #         }},
                {
                    SVC(): {
                        'kernel': ['poly', 'rbf', 'sigmoid'],
                        'C': [0.01, 0.1, 1, 10, 100],
                        'degree': [2, 3, 4, 5]
                        }},
                # {   
                #     AdaBoostClassifier(): { 
                #         'n_estimators': [100, 500, 1000, 5000]
                #         }},
                # {
                #     RandomForestClassifier(): { 
                #         'n_estimators': [50, 100, 150, 200],
                #         'criterion': ["gini", "entropy", "log_loss"],
                #         "max_features": ["auto", "sqrt", "log2"],
                #         "min_samples_leaf": [5, 10, 20, 50, 100],
                #         "max_depth": range(2, 20, 3)   
                #         }},
                # {
                #     GradientBoostingClassifier(): { 
                #         'n_estimators': [100, 500, 1000, 5000],
                #         'max_depth': range(5,16,2), 
                #         'min_samples_split': range(200,2100,200),
                #         'min_samples_leaf': range(30,71,10),
                #         'max_features': range(7,20,2),
                #         'subsample': [0.6,0.7,0.75,0.8,0.85,0.9]
                #         }},
                # {
                #     KNeighborsClassifier(): { 
                #         'n_neighbors': [2, 5, 7, 9, 11, 13, 15, 30, 60],
                #         'weights': ['uniform', 'distance'],
                #         'metric': ['minkowski', 'euclidean', 'manhattan'],
                #         "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                #         }},
                # {
                #     DecisionTreeClassifier(): {
                #         'criterion': ["gini", "entropy", "log_loss"],
                #         "max_features": ["auto", "sqrt", "log2"],
                #         "min_samples_leaf": [5, 10, 20, 50, 100],
                #         "max_depth": [2, 3, 5, 10, 20]    
                #         }}
                )
            
            best_model, model_report_val = self.utils.evaluate_models(models, X_train, y_train, X_val, y_val)

            logging.info("Best Model Report on Validation Dataset: {}".format(model_report_val))
            
            self.utils.save_object(
                 file_path=self.model_trainer_config.trained_model_obj_path,
                 obj=best_model
            )

            # model_report_test = self.utils.predict(best_model, X_test, y_test)
            # logging.info("Best Model Report on Test Dataset: {}".format(model_report_test))          

        except Exception as e:
            raise CustomException(e, sys)

    def show_model_score():
        pass
