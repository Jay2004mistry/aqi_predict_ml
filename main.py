#create fast api
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle,json
import pandas as pd
import numpy as np

app = FastAPI()
#use middleware to allow all origins, methods and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#load the model
model=pickle.load(open('aqi_model.pkl', 'rb'))
le=pickle.load(open('label_encoder.pkl', 'rb'))
feature_coulumns=json.load(open('feature_columns.json', 'r'))

# mapping for season
season_mapping = {
    'Winter': 0,
    'Summer': 1,
    'Monsoon': 2,
    'Post-Monsoon': 3
}
#mapping for AQI category labels
study_mapping = {0: 'Good', 1: 'Satisfactory', 2: 'Moderate', 3: 'Poor', 4: 'Very Poor', 5: 'Severe'}

#define the request body means the input data format for the prediction
class AQIRequest(BaseModel):
    state:str
    location:str
    type:str
    so2:float
    no2:float
    rspm:float
    spm:float
    year:int
    month:int
    season:str

@app.post('/predict')
#define the predict endpoint which takes the input data in the form of AQIRequest and returns the predicted AQI category and confidence score
def predict_aqi(request:AQIRequest):
    input_data = pd.DataFrame([{
        'state': request.state,
        'location': request.location,
        'type': request.type,
        'so2': request.so2,
        'no2': request.no2,
        'rspm': request.rspm,
        'spm': request.spm,
        'year': request.year,
        'month': request.month,
        'season': season_mapping.get(request.season, -1)  # Convert season to numeric
    }])[feature_coulumns]  # Ensure the order of columns matches the training data
    
    #make prediction in array and convert it into encoded label and get the confidence score like 
    #if model.predict() retuurn [3] so [0] get the first value->3. so predction=3(which means poor) 
    # and conf is the confidence score of the prediction
    prediction = model.predict(input_data)[0]

    #get the confidence score of the prediction by using predict_proba method which returns the probability of each class 
    # and we get the max probability as confidence score
    conf = model.predict_proba(input_data)[0].max()

#return the predicted AQI category and confidence score in percentage with 2 decimal places
    return {
        #predicted_aqi_category is the predicted AQI category based on the prediction
        "predicted_aqi_category": study_mapping.get(int(prediction), "Unknown"),
        #confidence tell us how confident the model is about its prediction. 
        "confidence": round(float(conf) * 100, 2)
    }


@app.get('/')
def home():
    return {"message": "Welcome to the AQI Prediction API. Use the /predict endpoint to get AQI category predictions."}
