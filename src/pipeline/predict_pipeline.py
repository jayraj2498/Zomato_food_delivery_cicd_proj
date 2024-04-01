import sys 
import os 
import pandas as pd 
from src.exception import CustomException 
from src.utils import load_object


class predictpipeline : 
    def __init__(self):
        pass 
    
    def predict(self,features):
        try:
            
            # preprocessor_path=os.path.join('artifacts','preprocessor.pkl')  do that 
            # model_path=os.path.join('artifacts','model.pkl')
            
            model_path=os.path.join("E:\\zomato_project\\artifacts\\model.pkl")
            preprocessor_path=os.path.join('E:\\zomato_project\\artifacts\\proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData :             # the class uis responsible for mapping all the input we are giving in html to backend with particular val  
     
    def __init__(self, 
        Delivery_person_Age: float,
        Delivery_person_Ratings: float,
        Weather_conditions: str,
        Road_traffic_density: str,
        Vehicle_condition: int,
        Type_of_order: str,
        Type_of_vehicle: str,
        multiple_deliveries: float,
        Festival: str,
        City: str,
        Time_Orderd_minute: float,
        Time_Orderd_second: float,
        Time_Order_picked_minute: float,
        Time_Order_picked_second: float):

        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Festival = Festival
        self.City = City
        self.Time_Orderd_minute = Time_Orderd_minute
        self.Time_Orderd_second = Time_Orderd_second
        self.Time_Order_picked_minute = Time_Order_picked_minute
        self.Time_Order_picked_second = Time_Order_picked_second
        
        
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
            "Delivery_person_Age": [self.Delivery_person_Age],
            "Delivery_person_Ratings": [self.Delivery_person_Ratings],
            "Weather_conditions": [self.Weather_conditions],
            "Road_traffic_density": [self.Road_traffic_density],
            "Vehicle_condition": [self.Vehicle_condition],
            "Type_of_order": [self.Type_of_order],
            "Type_of_vehicle": [self.Type_of_vehicle],
            "multiple_deliveries": [self.multiple_deliveries],
            "Festival": [self.Festival],
            "City": [self.City],
            "Time_Orderd_minute": [self.Time_Orderd_minute],
            "Time_Orderd_second": [self.Time_Orderd_second],
            "Time_Order_picked_minute": [self.Time_Order_picked_minute],
            "Time_Order_picked_second": [self.Time_Order_picked_second],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
        
        
        
# these funct return input in the form of dataframe 
# inshort from our  web application  whatever input is comming same input we are get mapp with the our abouve particaular val 