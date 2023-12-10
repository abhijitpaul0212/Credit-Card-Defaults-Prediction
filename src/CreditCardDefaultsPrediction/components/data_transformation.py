# data_transformation.py

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.CreditCardDefaultsPrediction.logger import logging
from src.CreditCardDefaultsPrediction.exception import CustomException
from src.CreditCardDefaultsPrediction.utils.utils import Utils
from src.CreditCardDefaultsPrediction.utils.data_processor import CSVProcessor
from src.CreditCardDefaultsPrediction.utils.transformer import UpperBoundCalculator, ClipTransformer, PositiveTransformer, OutlierTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, LabelEncoder

import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataTransformationConfig:
    """
    This is configuration class for Data Transformation
    """
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    This class handles Data Transformation
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.utils = Utils()
        self.csv_processor = CSVProcessor()

    def transform_data(self):
        try:
            numerical_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT_SEPT', 'BILL_AMT_AUG', 'BILL_AMT_JUL', 'BILL_AMT_JUN', 'BILL_AMT_MAY', 'BILL_AMT_APR',
                                  'PAY_AMT_SEPT', 'PAY_AMT_AUG', 'PAY_AMT_JUL', 'PAY_AMT_JUN', 'PAY_AMT_MAY', 'PAY_AMT_APR']

            categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_SEPT', 'PAY_AUG', 'PAY_JUL', 'PAY_JUN', 'PAY_MAY', 'PAY_APR']

            num_pipeline = Pipeline(
                steps=[
                    # ('upper_bound_calculator', UpperBoundCalculator()),
                    # ('feature_clip', ClipTransformer()),
                    # ('positive_transform', PositiveTransformer()),
                    # ('power_box_cox', PowerTransformer(method='box-cox', standardize=True)),
                    # ('outlier', OutlierTransformer()),
                    ('scaler', StandardScaler()),
                    
                ])

            cat_pipeline = Pipeline(
                steps=[
                    ('onehotencoder', OneHotEncoder(sparse_output=False,handle_unknown='ignore', categories='auto',drop='first')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])

            return preprocessor
        
        except Exception as e:
            logging.error("Exception occured in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        def replace_categories(df):
            df = df.replace({'SEX': {1: 'MALE', 2: 'FEMALE'},
                        'EDUCATION': {1: 'graduate_school', 2: 'university', 3: 'high_school', 4: 'others'},
                        'MARRIAGE': {1: 'married', 2: 'single', 3: 'others'}})
            logging.info("Numerical categories has been converted to string values")
            return df
        
        def modify_columns(df):
            # Modify 'EDUCATION' column
            fil_education = (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6) | (df['EDUCATION'] == 0)
            df.loc[fil_education, 'EDUCATION'] = 4

            # Modify 'MARRIAGE' column
            fil_marriage = df['MARRIAGE'] == 0
            df.loc[fil_marriage, 'MARRIAGE'] = 3

            logging.info("EDUCATION & MARRIAGE column's values are merged which has lesser counts")
            return df
    
        try:
            train_df = self.utils.run_data_pipeline(self.csv_processor, path=None, filename=train_path)
            test_df = self.utils.run_data_pipeline(self.csv_processor, path=None, filename=test_path)
            
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            # Handle imbalance data
            train_df = train_df.drop(columns=['_id'], axis=1)
            train_df = self.utils.smote_balance(train_df)

            test_df = test_df.drop(columns=['_id'], axis=1)
            test_df = self.utils.smote_balance(test_df)
            
            # Modify column data
            # train_df = modify_columns(train_df)
            # test_df = modify_columns(test_df)

            # Replace categories
            # train_df = replace_categories(train_df)
            # test_df = replace_categories(test_df)
            
            target_column_name = 'DEFAULT_PAYMENT'
            drop_columns = [target_column_name, 'ID']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply transformation
            preprocessing_obj = self.transform_data()
            preprocessing_obj.fit(input_feature_train_df)
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            input_feature_train_arr_df = pd.DataFrame(input_feature_train_arr, columns=preprocessing_obj.get_feature_names_out())
            input_feature_test_arr_df = pd.DataFrame(input_feature_test_arr, columns=preprocessing_obj.get_feature_names_out())

            logging.info("Applying preprocessing object on training, vdalidation and testing datasets")

            train_df = pd.concat([input_feature_train_arr_df, target_feature_train_df], axis=1)
            test_df = pd.concat([input_feature_test_arr_df, target_feature_test_df], axis=1)

            logging.info(f'Processed Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Processed Test Dataframe Head : \n{test_df.head().to_string()}')

            self.utils.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_df,
                test_df
            )

        except Exception as e:
            logging.error("Exception occured in Initiate Data Transformation")
            raise CustomException(e, sys)


if __name__ == '__main__':
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path="artifacts/train.csv", test_path="artifacts/test.csv")
