# prediction_pipeline.py

import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.CreditCardDefaultsPrediction.logger import logging
from src.CreditCardDefaultsPrediction.exception import CustomException
from src.CreditCardDefaultsPrediction.utils.utils import Utils


class PredictPipeline:
    def __init__(self):
        self.utils = Utils()

    def predict(self, features):
        try:
            logging.info('Prediction Pipeline initiated')
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = self.utils.load_object(preprocessor_path)
            model = self.utils.load_object(model_path)

            # preprocessed and scaled data
            scaled_data = preprocessor.transform(features)
            
            pred = model.predict(scaled_data)
            logging.info('Predicted value: {}'.format(pred))
            
            return pred

        except Exception as e:
            raise CustomException(e, sys)
  

class CustomData:
    def __init__(self,
                 limit_balance: float,
                 sex: int,
                 education: int,
                 marriage: int,
                 age: int,
                 pay_sept: int,
                 pay_aug: int,
                 pay_jul: int,
                 pay_jun: int,
                 pay_may: int,
                 pay_apr: int,
                 bill_amount_sept: float,
                 bill_amount_aug: float,
                 bill_amount_jul: float,
                 bill_amount_jun: float,
                 bill_amount_may: float,
                 bill_amount_apr: float,
                 pay_amount_sept: float,
                 pay_amount_aug: float,
                 pay_amount_jul: float,
                 pay_amount_jun: float,
                 pay_amount_may: float,
                 pay_amount_apr: float,):
        
        self.limit_balance = limit_balance
        self.sex = sex
        self.education = education
        self.marriage = marriage
        self.age = age
        self.pay_sept = pay_sept
        self.pay_aug = pay_aug
        self.pay_jul = pay_jul
        self.pay_jun = pay_jun
        self.pay_may = pay_may
        self.pay_apr = pay_apr
        self.bill_amount_sept = bill_amount_sept
        self.bill_amount_aug = bill_amount_aug
        self.bill_amount_jul = bill_amount_jul
        self.bill_amount_jun = bill_amount_jun
        self.bill_amount_may = bill_amount_may
        self.bill_amount_apr = bill_amount_apr
        self.pay_amount_sept = pay_amount_sept
        self.pay_amount_aug = pay_amount_aug
        self.pay_amount_jul = pay_amount_jul
        self.pay_amount_jun = pay_amount_jun
        self.pay_amount_may = pay_amount_may
        self.pay_amount_apr = pay_amount_apr
                
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'LIMIT_BAL': [self.limit_balance],
                'SEX': [self.sex],
                'EDUCATION': [self.education],
                'MARRIAGE': [self.marriage],
                'AGE': [self.age],
                'PAY_SEPT': [self.pay_sept],
                'PAY_AUG': [self.pay_aug],
                'PAY_JUL': [self.pay_jul],
                'PAY_JUN': [self.pay_jun],
                'PAY_MAY': [self.pay_may],
                'PAY_APR': [self.pay_apr],
                'BILL_AMT_SEPT': [self.bill_amount_sept],
                'BILL_AMT_AUG': [self.bill_amount_aug],
                'BILL_AMT_JUL': [self.bill_amount_jul],
                'BILL_AMT_JUN': [self.bill_amount_jun],
                'BILL_AMT_MAY': [self.bill_amount_may],
                'BILL_AMT_APR': [self.bill_amount_apr],
                'PAY_AMT_SEPT': [self.pay_amount_sept],
                'PAY_AMT_AUG': [self.pay_amount_aug],
                'PAY_AMT_JUL': [self.pay_amount_jul],
                'PAY_AMT_JUN': [self.pay_amount_jun],
                'PAY_AMT_MAY': [self.pay_amount_may],
                'PAY_AMT_APR': [self.pay_amount_apr],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Custom input is converted to Dataframe: \n{}'.format(df.head()))
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)
