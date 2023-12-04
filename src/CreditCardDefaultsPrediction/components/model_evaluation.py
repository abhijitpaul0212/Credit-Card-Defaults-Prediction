# model_evaluation.py

import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from urllib.parse import urlparse
from dataclasses import dataclass

from src.CreditCardDefaultsPrediction.logger import logging
from src.CreditCardDefaultsPrediction.exception import CustomException
from src.CreditCardDefaultsPrediction.utils.utils import Utils


@dataclass
class ModelEvaluation:

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        return accuracy, f1, precision, recall, roc_auc
    
    def initiate_model_evaluation(self, test_array):
        try:
            X_test, y_test = (test_array.iloc[:, :-1], test_array.iloc[:, -1])
            model_path = os.path.join("artifacts", "model.pkl")
            print(model_path)
            model = Utils().load_object(model_path)
            print(model)

            """
            If MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME & MLFLOW_TRACKING_PASSWORD are set correctly, 
            then tracking_url_type = https  --> DagsHub MLFlow
            else tracking_url_type = file --> Local MLFLow
            """
            mlflow.set_registry_uri("https://dagshub.com/abhijitpaul0212/CreditCardDefaultsPrediction.mlflow")
            tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type)

            with mlflow.start_run():
                predicted_qualities = model.predict(X_test)

                (accuracy, f1, precision, recall, roc_auc) = self.eval_metrics(actual=y_test, pred=predicted_qualities)
                self.eval_metrics(actual=y_test, pred=predicted_qualities)

                mlflow.log_metric("accuracy_score", accuracy)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision_score", precision)
                mlflow.log_metric("recall_score", recall)
                mlflow.log_metric("roc_auc_score", roc_auc)

                if tracking_url_type != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            raise CustomException(e, sys)
