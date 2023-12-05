# app.py
import streamlit as st
import pandas as pd
import os

from src.CreditCardDefaultsPrediction.pipelines.prediction_pipeline import PredictPipeline, CustomData
from src.CreditCardDefaultsPrediction.utils.utils import Utils
from src.CreditCardDefaultsPrediction.utils.data_processor import CSVProcessor
from src.CreditCardDefaultsPrediction.components.model_trainer import ModelTrainer


data = Utils().run_data_pipeline(CSVProcessor(), "notebooks/data/raw_data", "UCI_Credit_Card_Defaults.csv", skiprows=1, skipinitialspace=True)

data.drop(['default payment next month'], axis=1, inplace=True)

st.write("""
# Credit Card Defaults Prediction App

This app predicts the **Credit Card Defaults**!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


def user_input_features(data):
    limit_balance = st.number_input("LIMIT BALANCE:")

    # Dropdown to select text option for SEX
    sex_dict = {'Male': 1, 
                'Female': 2}
    sex = st.selectbox('SEX:', list(sex_dict.keys()))

    # Dropdown to select text option for EDUCATION
    education_dict = {'Graduate School': 1, 
                      'University': 2, 
                      'High School': 3, 
                      'Others': 4}
    education = st.selectbox('EDUCATION:', list(education_dict.keys()))

    # Dropdown to select text option for MARRIAGE
    marriage_dict = {'Married': 1, 
                     'Single': 2,
                     'Others': 3}
    marriage = st.selectbox('MARRIAGE:', list(marriage_dict.keys()))

    age = st.sidebar.slider('AGE', int(data['AGE'].min()), int(data['AGE'].max()), int(data['AGE'].mean()))

    # Dropdown to select text option for PAY SEPT
    pay_sept_dict = {'No Consumption': -2, 
                     'Pay Duly': -1, 
                     'Revolving': 0,  
                     'Delayed for 1 month': 1, 
                     'Delayed for 2 months': 2, 
                     'Delayed for 3 months': 3, 
                     'Delayed for 4 months': 4, 
                     'Delayed for 5 months': 5, 
                     'Delayed for 6 months': 6, 
                     'Delayed for 7 months': 7, 
                     'Delayed for 8 months': 8, 
                     'Delayed for 9 months and above': 9}
    pay_sept = st.selectbox('PAY SEPTEMBER:', list(pay_sept_dict.keys()))

    # Dropdown to select text option for PAY AUG
    pay_aug_dict = {'No Consumption': -2, 
                    'Pay Duly': -1, 
                    'Revolving': 0,  
                    'Delayed for 1 month': 1, 
                    'Delayed for 2 months': 2, 
                    'Delayed for 3 months': 3, 
                    'Delayed for 4 months': 4, 
                    'Delayed for 5 months': 5, 
                    'Delayed for 6 months': 6, 
                    'Delayed for 7 months': 7, 
                    'Delayed for 8 months': 8, 
                    'Delayed for 9 months and above': 9}
    pay_aug = st.selectbox('PAY AUGUST:', list(pay_aug_dict.keys()))

    # Dropdown to select text option for PAY JUL
    pay_jul_dict = {'No Consumption': -2,
                    'Pay Duly': -1, 
                    'Revolving': 0,  
                    'Delayed for 1 month': 1, 
                    'Delayed for 2 months': 2, 
                    'Delayed for 3 months': 3, 
                    'Delayed for 4 months': 4, 
                    'Delayed for 5 months': 5, 
                    'Delayed for 6 months': 6, 
                    'Delayed for 7 months': 7, 
                    'Delayed for 8 months': 8, 
                    'Delayed for 9 months and above': 9}
    pay_jul = st.selectbox('PAY JULY:', list(pay_jul_dict.keys()))

    # Dropdown to select text option for PAY JUN
    pay_jun_dict = {'No Consumption': -2, 
                    'Pay Duly': -1, 
                    'Revolving': 0,  
                    'Delayed for 1 month': 1, 
                    'Delayed for 2 months': 2, 
                    'Delayed for 3 months': 3, 
                    'Delayed for 4 months': 4, 
                    'Delayed for 5 months': 5, 
                    'Delayed for 6 months': 6, 
                    'Delayed for 7 months': 7, 
                    'Delayed for 8 months': 8, 
                    'Delayed for 9 months and above': 9}
    pay_jun = st.selectbox('PAY JUNE:', list(pay_jun_dict.keys()))

    # Dropdown to select text option for PAY MAY
    pay_may_dict = {'No Consumption': -2, 
                    'Pay Duly': -1, 
                    'Revolving': 0,  
                    'Delayed for 1 month': 1, 
                    'Delayed for 2 months': 2, 
                    'Delayed for 3 months': 3, 
                    'Delayed for 4 months': 4, 
                    'Delayed for 5 months': 5, 
                    'Delayed for 6 months': 6, 
                    'Delayed for 7 months': 7, 
                    'Delayed for 8 months': 8, 
                    'Delayed for 9 months and above': 9}
    pay_may = st.selectbox('PAY MAY:', list(pay_may_dict.keys()))

    # Dropdown to select text option for PAY APR
    pay_apr_dict = {'No Consumption': -2, 
                    'Pay Duly': -1, 
                    'Revolving': 0,  
                    'Delayed for 1 month': 1, 
                    'Delayed for 2 months': 2, 
                    'Delayed for 3 months': 3, 
                    'Delayed for 4 months': 4, 
                    'Delayed for 5 months': 5, 
                    'Delayed for 6 months': 6, 
                    'Delayed for 7 months': 7, 
                    'Delayed for 8 months': 8, 
                    'Delayed for 9 months and above': 9}
    pay_apr = st.selectbox('PAY APRIL:', list(pay_apr_dict.keys()))

    # TextField to enter Bill AMOUNTS
    bill_amount_sept = st.number_input("BILL AMOUNT SEPTEMBER:")
    bill_amount_aug = st.number_input("BILL AMOUNT AUGUST:")
    bill_amount_jul = st.number_input("BILL AMOUNT JULY:")
    bill_amount_jun = st.number_input("BILL AMOUNT JUNE:")
    bill_amount_may = st.number_input("BILL AMOUNT MAY:")
    bill_amount_apr = st.number_input("BILL AMOUNT APRIL:")

    # TextField to enter PAID AMOUNTS
    pay_amount_sept = st.number_input("PAY AMOUNT SEPTEMBER:")
    pay_amount_aug = st.number_input("PAY AMOUNT AUGUST:")
    pay_amount_jul = st.number_input("PAY AMOUNT JULY:")
    pay_amount_jun = st.number_input("PAY AMOUNT JUNE:")
    pay_amount_may = st.number_input("PAY AMOUNT MAY:")
    pay_amount_apr = st.number_input("PAY AMOUNT APRIL:")

    data = {
                'LIMIT_BAL': float(limit_balance),
                'SEX': sex_dict[sex] if sex in sex_dict else 0,
                'EDUCATION': education_dict[education] if education in education_dict else 0,
                'MARRIAGE': marriage_dict[marriage] if marriage in marriage_dict else 0,
                'AGE': int(age),
                'PAY_0': pay_sept_dict[pay_sept] if pay_sept in pay_sept_dict else 0,
                'PAY_2': pay_aug_dict[pay_aug] if pay_aug in pay_aug_dict else 0,
                'PAY_3': pay_jul_dict[pay_jul] if pay_jul in pay_jul_dict else 0,
                'PAY_4': pay_jun_dict[pay_jun] if pay_jun in pay_jun_dict else 0,
                'PAY_5': pay_may_dict[pay_may] if pay_may in pay_may_dict else 0,
                'PAY_6': pay_apr_dict[pay_apr] if pay_apr in pay_apr_dict else 0,
                'BILL_AMT1': float(bill_amount_sept),
                'BILL_AMT2': float(bill_amount_aug),
                'BILL_AMT3': float(bill_amount_jul),
                'BILL_AMT4': float(bill_amount_jun),
                'BILL_AMT5': float(bill_amount_may),
                'BILL_AMT6': float(bill_amount_apr),
                'PAY_AMT1': float(pay_amount_sept),
                'PAY_AMT2': float(pay_amount_aug),
                'PAY_AMT3': float(pay_amount_jul),
                'PAY_AMT4': float(pay_amount_jun),
                'PAY_AMT5': float(pay_amount_may),
                'PAY_AMT6': float(pay_amount_apr)
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features(data)

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(df)

st.header('Customer will')
st.write("DEFAULT" if prediction[0] == 1 else "NOT DEFAULT")

st.write('---')
