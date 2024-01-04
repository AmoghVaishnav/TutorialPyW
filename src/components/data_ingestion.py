import os 
import sys
from typing import Any
from src.exception import Customexception
from src.logger import logging
import pandas as pd 

from sklearn.model_selection import train_test_split 
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train_csv")
    test_data_parh: str=os.path.join('artifacts',"test_csv")
    raw_data_path: str=os.path.join('artifacts',"data_csv")

class DataIngestion:
    def __call__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enetred the data ingestion method")
        try:
            df=pd.read_csv('TutorialPyW\ingestion\stud.csvv')
            logging.info("Data read successfully as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            
            logging.info("train and test split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info("Ingestion of data is complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise Customexception(e,sys)