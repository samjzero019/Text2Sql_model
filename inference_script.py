# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:37:00 2024

@author: shahs10
"""

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


def main():
    
    data = ["Passport", "Married", 'Male', "Governmental", 550]

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
    
    
    return list_recommended_cars