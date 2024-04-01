

from src.exception import CustomException 
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass   


from src.components.data_ingestion  import DataIngestionConfig
from src.components.data_ingestion  import DataIngestion


from src.components.data_transformation  import DataTransformationConfig 
from src.components.data_transformation  import DataTransformation 

# from src.components.model_trainer import DataIngestionConfig
from src.components.model_trainer import ModelTrainerConfig 
from src.components.model_trainer import ModelTrainer



if __name__=="__main__":
    obj=DataIngestion()
    train_data ,test_data =obj.initiate_data_ingestion() 
    
    data_transformation=DataTransformation() 
    train_arr ,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)  # we are giving data 

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))