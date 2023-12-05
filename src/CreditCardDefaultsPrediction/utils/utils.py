# utils.py
import pickle
import sys
import os
import pandas as pd
from time import time
from src.CreditCardDefaultsPrediction.exception import CustomException
from src.CreditCardDefaultsPrediction.logger import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE


# libraries to save pickle
import joblib


class Utils:

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

    def predict(self, model, features, label):
        MODEL_REPORT = {}

        pred_label = model.predict(features)
        logging.info("Model prediction completed")
        
        logging.info("Confusion Matrix: \n{}".format(confusion_matrix(y_true=label, y_pred=pred_label)))
        
        MODEL_REPORT['Model'] = model
        MODEL_REPORT['Accuracy Score'] = accuracy_score(y_true=label, y_pred=pred_label)
        MODEL_REPORT['F1 Score'] = f1_score(y_true=label, y_pred=pred_label)
        MODEL_REPORT['Precision Score'] = precision_score(y_true=label, y_pred=pred_label)
        MODEL_REPORT['Recall Score'] = recall_score(y_true=label, y_pred=pred_label)
        MODEL_REPORT['ROC AUC Score'] = roc_auc_score(y_true=label, y_score=pred_label)
        
        # logging.info("Best Model Report: {}".format(MODEL_REPORT))
        return MODEL_REPORT

    def evaluate_models(self, models: tuple, train_features, train_label, val_features, val_label):

        def find_model_by_score(dictionary, target_value):
            for key, value in dictionary.items():
                if value == target_value:
                    return key
            return None
        
        SCORING_METRIC = "recall"
        TRAINING_SCORE = {}
        for items in models:
            for model, param in items.items():
                
                model_name = str(model).split("()")[0]
                logging.info("\n\n========================= {} =======================".format(model_name))
                start = time()
                cv = RandomizedSearchCV(estimator=model, param_distributions=param, cv=5, n_jobs=-1, scoring=SCORING_METRIC)
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
        model_report = self.predict(model=best_model, features=val_features, label=val_label)

        return (best_model, model_report)

    def smote_balance(self, data):
        target_column_name = 'default payment next month'
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        
        logging.info('Dataset shape prior resampling: {}'.format(data.shape[0]))
        X_resampled, y_resampled = sm.fit_resample(X=data.drop(columns=target_column_name), y=data[target_column_name])
        data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
        logging.info('Dataset shape after resampling: {}'.format(data.shape[0]))
        return data

    def connect_database(self, uri):
        # making connection with mongo db
        from pymongo.mongo_client import MongoClient
        # uri = "mongodb+srv://root:root@cluster0.k3s4vuf.mongodb.net/?retryWrites=true&w=majority&ssl=true"

        # Create a new client and connect to the server
        client = MongoClient(uri)

        # Send a ping to confirm a successful connection
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            return client
        except Exception as e:
            logging.error("Exception occured during creating database connection")
            raise CustomException(e, sys)

    def get_data_from_database(self, uri, collection):
        # collection = "credit_card_defaults/data"
        collection = collection.split("/")
        client = self.connect_database(uri)
        collection = client[collection[0]][collection[1]]
        data = list(collection.find())
        return pd.DataFrame(data)


if __name__ == "__main__":
    logging.info("Demo logging activity")

    utils = Utils()
    utils.save_object(os.path.join('logs', 'utils.pkl'), utils)
    utils.load_object(os.path.join('logs', 'utils.pkl'))
    utils.delete_object(os.path.join('logs', 'utils.pkl'))
