# data_ingestion.py

import sys
import os
from dataclasses import dataclass

from src.CreditCardDefaultsPrediction.logger import logging
from src.CreditCardDefaultsPrediction.exception import CustomException
from src.CreditCardDefaultsPrediction.utils.data_processor import CSVProcessor, DBProcessor
from src.CreditCardDefaultsPrediction.utils.utils import Utils

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataIngestionConfig:
    """
    This is configuration class for Data Ingestion
    """
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    val_data_path: str = os.path.join("artifacts", "val.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """
    This class handled Data Ingestion
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.utils = Utils()
        # self.csv_processor = CSVProcessor()
        self.db_processor = DBProcessor()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        try:
            # Read raw dataset directly from csv file
            # data = self.utils.run_data_pipeline(self.csv_processor, "notebooks/data/raw_data", "UCI_Credit_Card_Defaults.csv", skiprows=1, skipinitialspace=True)

            # Read raw dataset from MongoDB database
            data = self.utils.run_data_pipeline(self.db_processor, "mongodb+srv://root:root@cluster0.k3s4vuf.mongodb.net/?retryWrites=true&w=majority&ssl=true", "credit_card_defaults/data")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw dataset is saved in artifacts folder")

            train_data, test_data = train_test_split(data, test_size=0.40, random_state=42)
            val_data, test_data = train_test_split(test_data, test_size=0.50, random_state=42)
            logging.info("Dataset is splitted into Train, Validation & Test data")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            val_data.to_csv(self.ingestion_config.val_data_path, index=False)
            logging.info("Train, Test & validation dataset are saved in artifacts folder")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Exception occuring during data ingestion")
            raise CustomException(e, sys)


if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data_path, val_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    print(train_data_path)
    print(val_data_path)
    print(test_data_path)
