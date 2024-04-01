import sys
import os 
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation  opearation 
        where we handle null val  , ohe , standard_scaler by using pipeline 
        
        '''
        try:
            numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
       'multiple_deliveries', 'Time_Orderd_minute', 'Time_Orderd_second',
       'Time_Order_picked_minute', 'Time_Order_picked_second']
            categorical_columns = [
                'Weather_conditions', 'Road_traffic_density', 'Type_of_order',
                'Type_of_vehicle', 'Festival', 'City',
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),  # here we handle null val if it present 
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)   
        
        
    
    def initiate_data_transformation(self,train_path,test_path): 
        
        try : 
            train_df = pd.read_csv(train_path) 
            test_df = pd.read_csv(test_path)   
            
            logging.info(" reading the test data process is complted ")  
            
            logging.info(" obtaining preprocesing object " )   
            
            preprocessing_obj=self.get_data_transformer_object()
            
            
            target_column_name="Time_taken (min)" 
            numerical_columns  =  ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
                                    'multiple_deliveries', 'Time_Orderd_minute', 'Time_Orderd_second',
                                    'Time_Order_picked_minute', 'Time_Order_picked_second']
            
            
            input_feature_train_df = train_df.drop(columns=[target_column_name] , axis=1)  
            target_feature_train_df = train_df[target_column_name]   
            
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]  
            
            
            logging.info(
                f" We Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            # we are apply  standard scaler and all method 
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)  
            
            # we combine train_test data in one form 
            train_arr = np.c_[input_feature_train_arr , np.array(target_feature_train_df)]   
            test_arr = np.c_[input_feature_test_arr , np.array(target_feature_test_df)]   
            
            
            logging.info(f" We Applyed all preprocesing object steps ")   
            
            
            save_object(self.data_transformation_config.preprocessor_obj_file_path  ,  # here we are saving pickle name is harddisk   
                        obj=preprocessing_obj)
            
            
            return(
                train_arr , 
                test_arr , 
                self.data_transformation_config.preprocessor_obj_file_path
                
            )
            
            
        except Exception as e :
            raise CustomException(e,sys)