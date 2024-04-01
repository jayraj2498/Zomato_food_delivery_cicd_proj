# utils have all common functonality that entire programe can use 
import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)    
    
    
    
def evaluate_models(X_train , y_train , X_test , y_test , models ,params) : 
    try:
        report={}
        
        for i in range(len(list(models))) :
            model= list(models.values())[i]             # range of every model 
            para=params[list(models.keys())[i]]         # range of evry parameters 
            
            gs=GridSearchCV(model,para , cv=4)               
            gs.fit(X_train,y_train)  
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)  
            
            y_train_pred=model.predict(X_train)
            y_test_pred= model.predict(X_test) 
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred) 
            
            report[list(models.keys())[i]] = test_model_score 
            
        return report 
         
    except Exception as e  :
        raise CustomException(e,sys)   
    
    
    

   # it is opeing file path in read_bite mode and it is opening and it load pickle file by using dill (the load object it is responsible for loading pickle file )
    

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

       
    