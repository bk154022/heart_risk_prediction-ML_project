import os
import sys
sys.path.append('E:/Ai_projects/Heart_risk_prediction/src')
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","train.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")

class Dataingestion:
    def __init__(self):
        self.ingestionConfig=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or components")
        try:
            df=pd.read_csv("E:\\Ai_projects\\Heart_risk_prediction\\Notebook\\heart_attack_prediction_dataset.csv")
            logging.info("read the dataset as DataFrame")
            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path),exist_ok=True)
            df.to_csv(self.ingestionConfig.raw_data_path,index=False,header=True)
            logging.info("trian_test_split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestionConfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionConfig.test_data_path,index=False,header=True)
            logging.info("ingestion of data is completed")

            return self.ingestionConfig.train_data_path, self.ingestionConfig.test_data_path
        except Exception as e :
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=Dataingestion()
    obj.initiate_data_ingestion()