# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import subprocess
import json

# custom 
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import joblib
#
 

def apply_saved_encoder(encoder_filename, df, column):
    encoder = joblib.load(encoder_filename)

    encoded_data = encoder.transform(df[[column]])

    # Use the feature names from the encoder as column names
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column]))

    # Concatenate the original DataFrame with the encoded DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    # Drop the original column from the DataFrame
    df = df.drop([column], axis=1)

    return df


    
def normalize_columns(df, column_name, min_val=300, max_val=850):
    data = df[column_name].values.reshape(-1, 1)
    
    
    # Add a new normalized column to the DataFrame
    df['normalized_{}'.format(column_name)] = (df[column_name] - min_val) / (max_val - min_val)

    
    # Drop the original column
    df.drop(column_name, axis=1, inplace=True)
    
    return df




def process_result(result):
    top_classes = np.argsort(result[0])[::-1][:3]  # Indices of the top three classes
    top_probabilities = result[0, top_classes]  # Probabilities of the top three classes
    le = joblib.load('label_encoder.joblib')
    # Display the top three classes and their probabilities
    decoded_top_classes = le.inverse_transform(top_classes)
    return decoded_top_classes



def process_output(recommended_cars):
    transformed_data = []
    for car_name in recommended_cars:
        car_dict = {
            "CarMake": car_name.split()[0],
            "CarModel": car_name
        }
        transformed_data.append(car_dict)
    return transformed_data






## END 
app = FastAPI()


 
 
@app.get('/ping')
async def ping():
    return {"message": "ok"}
 
@app.on_event('startup')
def load_model():
    return "classifier"
 
@app.post('/invocations')
def invocations(request):
    print("Request",request)
    try:
        # Extract values from the request
        id_type = request.id_type
        marital_status = request.marital_status
        gender = request.gender
        employment_type = request.employment_type
        credit_score = request.credit_score
        
        # data = ["Passport", "Married", 'Male', "Governmental", 550]
        data = [id_type,marital_status, gender, employment_type, credit_score]
        column_names = ['IDType', 'BPKBOwnerMaritalStatus', 'Gender', 'EmploymentType', 'CreditScore']
        df = pd.DataFrame([data], columns=column_names)

        categorical_features = ['IDType', "BPKBOwnerMaritalStatus", 'Gender', "EmploymentType"]
        
        for feature in categorical_features:
        
            joblib_file = "{}.joblib".format(feature)
            df = apply_saved_encoder(joblib_file, df, feature)
        
        numerical_columns = ['CreditScore']
        for column in numerical_columns:
            print("COLUMN ", column)
            df = normalize_columns(df, column, min_val=300, max_val = 850)
        
        
        model = keras.models.load_model('model')
        
        result = model.predict(df)
        
        recommended_cars = process_result(result)
        
        list_recommended_cars = process_output(recommended_cars)
        
            
        return JSONResponse(content=list_recommended_cars, status_code=200)
    except Exception as e:
        # Handle exceptions and return an appropriate response
        error_message = f"An error occurred: {str(e)}"
        return HTTPException(status_code=500, detail=error_message)

    #return {"message": "Invocation Request Received"}
