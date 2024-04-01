# here we doing web application where we having form , 
# here we give all our input data that is required to presict students performances 
# here we considering using flask app 


from flask import Flask , request , render_template  
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler   
from src.pipeline.predict_pipeline import CustomData , predictpipeline 

application=Flask(__name__)
app=application 

@app.route('/') 
def index() :
    return render_template('index.html')


@app.route('/predictdata' , methods=['GET','POST']) 
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  
    else :                                                                      # in the post part we have to capture data , we have to do standarscaling then we do prediction
        data=CustomData(
            Delivery_person_Age = float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density'),
            Vehicle_condition = int(request.form.get('Vehicle_condition')),
            Type_of_order = request.form.get('Type_of_order'),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            multiple_deliveries = float(request.form.get('multiple_deliveries')),
            Festival = request.form.get('Festival'),
            City = request.form.get('City'),
            Time_Orderd_minute = float(request.form.get('Time_Orderd_minute')),
            Time_Orderd_second = float(request.form.get('Time_Orderd_second')),
            Time_Order_picked_minute = float(request.form.get('Time_Order_picked_minute')),
            Time_Order_picked_second = float(request.form.get('Time_Order_picked_second'))

        )
        
        pred_df=data.get_data_as_data_frame()                # we are converted our input data in to dataframe 
        print(pred_df)  
        print("before prediction")             
         
        predict_pipeline= predictpipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)                       # throght predict opreation we get our output(here we are using all power of of predict function from pipeline.py)  
        print("after Prediction")
        results=round(results[0],2)
        
        return render_template('results.html',final_result=results)         # our putput it is in the list format 
        
        
        
if __name__=="__main__":
    app.run(host="0.0.0.0" , port=5000)                   # debug=True  we will remove debug , while we deplye on cloud 
        
