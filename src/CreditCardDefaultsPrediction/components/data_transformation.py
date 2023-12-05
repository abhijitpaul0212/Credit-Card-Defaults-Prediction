# data_transformation.py

import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.CreditCardDefaultsPrediction.logger import logging
from src.CreditCardDefaultsPrediction.exception import CustomException
from src.CreditCardDefaultsPrediction.utils.utils import Utils
from src.CreditCardDefaultsPrediction.utils.data_processor import CSVProcessor
from src.CreditCardDefaultsPrediction.utils.transformer import UpperBoundCalculator, ClipTransformer, PositiveTransformer, OutlierTransformer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer

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
            numerical_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
            'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

            categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']


            num_pipeline = Pipeline(
                steps=[
                    # ('upper_bound_calculator', UpperBoundCalculator()),
                    # ('feature_clip', ClipTransformer()),
                    # ('positive_transform', PositiveTransformer()),
                    # ('power_box_cox', PowerTransformer(method='box-cox', standardize=True)),
                    ('outlier', OutlierTransformer()),
                    ('scaler', StandardScaler()),
                    
                ])

            cat_pipeline = Pipeline(
                steps=[
                    # ('imputer', SimpleImputer(strategy='most_frequent')),
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

    def initiate_data_transformation(self, train_path, val_path, test_path):
        try:
            train_df = self.utils.run_data_pipeline(self.csv_processor, path=None, filename=train_path)
            val_df = self.utils.run_data_pipeline(self.csv_processor, path=None, filename=val_path)
            test_df = self.utils.run_data_pipeline(self.csv_processor, path=None, filename=test_path)
                        
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Val Dataframe Head : \n{val_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            preprocessing_obj = self.transform_data()
            
            target_column_name = 'default payment next month'
            drop_columns = [target_column_name, 'ID']

            # Handle imbalance data

            train_df = train_df.drop(columns=['_id'], axis=1)
            train_df = self.utils.smote_balance(train_df)
            val_df = val_df.drop(columns=['_id'], axis=1)
            val_df = self.utils.smote_balance(val_df)
            test_df = test_df.drop(columns=['_id'], axis=1)
            test_df = self.utils.smote_balance(test_df)

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_val_df = val_df.drop(columns=drop_columns, axis=1)
            target_feature_val_df = val_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            preprocessing_obj.fit(input_feature_train_df)
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df)
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            input_feature_train_arr_df = pd.DataFrame(input_feature_train_arr, columns=input_feature_train_df.columns)
            input_feature_val_arr_df = pd.DataFrame(input_feature_val_arr, columns=input_feature_val_df.columns)
            input_feature_test_arr_df = pd.DataFrame(input_feature_test_arr, columns=input_feature_test_df.columns)

            logging.info("Applying preprocessing object on training, vdalidation and testing datasets")

            train_df = pd.concat([input_feature_train_arr_df, target_feature_train_df], axis=1)
            val_df = pd.concat([input_feature_val_arr_df, target_feature_val_df], axis=1)
            test_df = pd.concat([input_feature_test_arr_df, target_feature_test_df], axis=1)

            logging.info(f'Processed Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Processed Val Dataframe Head : \n{val_df.head().to_string()}')
            logging.info(f'Processed Test Dataframe Head : \n{test_df.head().to_string()}')

            self.utils.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_df,
                val_df,
                test_df
            )

        except Exception as e:
            logging.error("Exception occured in Initiate Data Transformation")
            raise CustomException(e, sys)


if __name__ == '__main__':
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path="artifacts/train.csv", val_path="artifacts/val.csv", test_path="artifacts/test.csv")