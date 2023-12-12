# utils.py
from cProfile import label
import pickle
import sys
import os
import pandas as pd
import numpy as np
from time import time
from pymongo.mongo_client import MongoClient
from src.CreditCardDefaultsPrediction.exception import CustomException
from src.CreditCardDefaultsPrediction.logger import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE


class Utils:

    def __init__(self) -> None:
        self.MODEL_REPORT = {}

    def save_object(self, file_path: str, obj):
        """
        The save_object function saves an object to a file.

        :param file_path: str: Specify the path where the object will be saved
        :param obj: obj: Pass the object to be saved
        :return: None
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
                logging.info(f"File is saved at '{file_path}' successfully.")
        except Exception as e:
            logging.error("Exception occured during saving object")
            raise CustomException(e, sys)

    def load_object(self, file_path: str):
        """
        The load_object function loads a pickled object from the file_path.

        :param file_path: str: Specify the path of the file that we want to load
        :return: None
        """
        try:
            with open(file_path, "rb") as file_obj:
                logging.info(f"File at '{file_path}' has been successfully loaded.")
                return pickle.load(file_obj)                
        except Exception as e:
            logging.error("Exception occured during loading object")
            raise CustomException(e, sys)

    def delete_object(self, file_path: str):
        """
        The delete_object function deletes a file from the local filesystem.
        
        :param file_path: str: Specify the path of the file to be deleted
        :return: None
        """
        try:
            # Check if the file exists
            if os.path.exists(file_path):
                # Remove the file
                os.remove(file_path)
                logging.info(f"File at '{file_path}' has been successfully deleted.")
            else:
                logging.info(f"File at '{file_path}' does not exist.")
        except Exception as e:
            logging.error("Exception occured during deleting object")
            raise CustomException(e, sys)

    def run_data_pipeline(self, data_processor, path: str = None, filename: str = None, **kwargs):
        """
        The run_data_pipeline function is a wrapper function that takes in the data_processor object and 
            calls its process_data method.
        
        :param data_processor: obj: Pass in the data processor class that will be used to process the data
        :param path: str: Specify the path to the data file
        :param filename: str: Specify the name of the file that will be used for processing
        :param **kwargs: Pass a variable number of keyword arguments to the function
        :return: The processed data
        """
        logging.info("Dataset loaded sucessfully")
        return data_processor.process_data(path, filename, **kwargs)

    def predict(self, model_name, model, features, label):
        """
        The predict function predicts the labels using the model and features provided.

        :param model: Pass the model to be used for prediction
        :param features: DataFrame: Features to be used for prediction
        :param label: DataFrame: Label to be used for prediction
        :return: dict: A dictionary with the model, accuracy score, f-score, precision score and recall score
        """
        pred_label = model.predict(features)
        logging.info("Model prediction completed")
        logging.info("Confusion Matrix: \n{}".format(confusion_matrix(y_true=label, y_pred=pred_label)))        
        self.MODEL_REPORT[model_name] = {
            'model': model,
            'accuracy': accuracy_score(y_true=label, y_pred=pred_label),
            'f1': f1_score(y_true=label, y_pred=pred_label),
            'precision': precision_score(y_true=label, y_pred=pred_label),
            'recall': recall_score(y_true=label, y_pred=pred_label),
            'roc-auc': roc_auc_score(y_true=label, y_score=pred_label)}

    def evaluate_models_with_hyperparameter(self, models: tuple, train_features, train_label, test_features, test_label, metric='accuracy'):
        """
        The evaluate_models function takes in a tuple of models and their parameters, 
        train_features, train_label, val_features and val_label. It then uses the RandomizedSearchCV function to find the best model for each model passed into it.
        
        :param models: tuple: Models and their parameters
        :param train_features: DataFrame: Training features to the evaluate_models function
        :param train_label: DataFrame: Trtaining labels to the predict function
        :param val_features: DataFrame: Validation features to the evaluate_models function
        :param val_label: Validation labels to the predict function
        :return: tuple: The best model and a dictionary of the model report
        """     
        def find_model_by_score(dictionary, target_value):
            for key, value in dictionary.items():
                if value == target_value:
                    return key
            return None

        np.random.seed(42)        
        TRAINING_SCORE = {}
        for items in models:
            for model, param in items.items():                
                model_name = str(model).split("()")[0]
                logging.info("\n\n========================= {} =======================".format(model_name))
                start = time()
                cv = GridSearchCV(estimator=model, param_grid=param, cv=3, n_jobs=-1, scoring=metric)
                cv.fit(train_features, train_label)
                end = time()
                logging.info("BEST PARAMS: {}".format(cv.best_params_))
                logging.info("BEST SCORE: {}".format(cv.best_score_))
                logging.info("Model took: {} secs".format(round(end-start, 4)))
                TRAINING_SCORE[cv.best_estimator_] = cv.best_score_

        logging.info("All training scores: {}".format(TRAINING_SCORE))

        best_score = sorted([value for key, value in TRAINING_SCORE.items()], reverse=True)[0]
        best_model = find_model_by_score(TRAINING_SCORE, best_score)

        logging.info("Model got trained")
        
        model_name = str(best_model).split("()")[0]
        self.predict(model_name=model_name, model=best_model, features=test_features, label=test_label)

        logging.info("BEST MODEL: {}".format(model_name))
        logging.info("TESTING SCORES: {}".format(self.MODEL_REPORT[model_name]))

        return best_model
    
    def evaluate_models(self, models: dict, train_features, train_label, test_features, test_label, metric='accuracy'):
        """
        The evaluate_models function takes in a tuple of models and their parameters, 
        train_features, train_label, val_features and val_label. It then uses the RandomizedSearchCV function to find the best model for each model passed into it.
        
        :param models: tuple: Models and their parameters
        :param train_features: DataFrame: Training features to the evaluate_models function
        :param train_label: DataFrame: Trtaining labels to the predict function
        :param val_features: DataFrame: Validation features to the evaluate_models function
        :param val_label: Validation labels to the predict function
        :return: tuple: The best model and a dictionary of the model report
        """     
        np.random.seed(42)        
        self.MODEL_REPORT = {}
        for model_name, model in models.items():            
            logging.info("\n\n========================= {} =======================".format(model_name))
            start = time()
            model.fit(train_features, train_label)
            end = time()
            logging.info("Model took: {} secs".format(round(end-start, 4)))

            # Evaluate the best model on the train & test set
            self.predict(model_name=model_name, model=model, features=test_features, label=test_label)
            
        logging.info("Model Report: {}".format(self.MODEL_REPORT))
        best_model_score = max(sorted(model[metric] for model in self.MODEL_REPORT.values()))
        best_model_name = list(self.MODEL_REPORT.keys())[list(model[metric] for model in self.MODEL_REPORT.values()).index(best_model_score)]
        best_model = self.MODEL_REPORT[best_model_name]['model']
        model_report = self.MODEL_REPORT[best_model_name]
        print("BEST MODEL REPORT: ", model_report)   
        return best_model

    def smote_balance(self, data):
        """
        The smote_balance function takes in a dataframe and returns the same dataframe with SMOTE resampling applied.
        
        :param data: DataFrame: Pass in the dataframe
        :return: DataFrame: Dataframe with the same number of rows as the original dataset, but now there are an equal number of 0s and 1s in the target column
        """
        
        target_column_name = 'DEFAULT_PAYMENT'
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        
        logging.info('Dataset shape prior resampling: {}'.format(data.shape[0]))
        X_resampled, y_resampled = sm.fit_resample(X=data.drop(columns=target_column_name), y=data[target_column_name])
        data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        logging.info('Dataset shape after resampling: {}'.format(data.shape[0]))
        return data

    def connect_database(self, uri):
        """
        The connect_database function establishes a connection to the MongoDB database.
        
        :param uri: str: URI of mongodb atlas database
        :return: MongoClient: A mongoclient object
        """       
        # uri = "mongodb+srv://root:root@cluster0.k3s4vuf.mongodb.net/?retryWrites=true&w=majority&ssl=true"

        client = MongoClient(uri)
        try:
            client.admin.command('ping')
            logging.info("MongoDb connection established successfully")
            return client
        except Exception as e:
            logging.error("Exception occured during creating database connection")
            raise CustomException(e, sys)

    def get_data_from_database(self, uri, collection):
        """
        The get_data_from_database function takes in a uri and collection name, connects to the database, 
        and returns a pandas dataframe of the data from that collection.

        :param uri: str: MongoDB database URI
        :param collection: str: Database name along with Collection e.g. "credit_card_defaults/data"
        :return: DataFrame: A pandas dataframe
        """
        collection = collection.split("/")
        client = self.connect_database(uri)
        collection = client[collection[0]][collection[1]]
        data = list(collection.find())
        return pd.DataFrame(data)
