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
model            = pickle.load(open('aqi_model.pkl', 'rb'))
#load 3 separate label encoders because we have 3 categorical columns state, location, type
#and each encoder is trained on different column so we need to load them separately
le_state         = pickle.load(open('le_state.pkl', 'rb'))
le_location      = pickle.load(open('le_location.pkl', 'rb'))
le_type          = pickle.load(open('le_type.pkl', 'rb'))
#load the feature columns to ensure the order of columns matches the training data
feature_coulumns = json.load(open('feature_columns.json', 'r'))

#we need mapping for all the categorical variables to convert them into numeric values 
# because our model is trained on numeric values and we need to convert the input data into numeric values before making predictions
#and we are taking value as string not numeric because user dont know the encodedvalue
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
    try:
        input_data = pd.DataFrame([{
            #convert state, location, type to number using label encoder
            'state':    le_state.transform([request.state])[0],
            'location': le_location.transform([request.location])[0],
            'type':     le_type.transform([request.type])[0],
            'so2':      request.so2,
            'no2':      request.no2,
            'rspm':     request.rspm,
            'spm':      request.spm,
            'year':     request.year,
            'month':    request.month,
            'season':   season_mapping.get(request.season, -1)  # Convert season to numeric
        }])[feature_coulumns]  # Ensure the order of columns matches the training data
    except ValueError as e:
        return {"error": f"Unknown value: {str(e)}. Use values from training data."}

    #make prediction in array and convert it into encoded label and get the confidence score like
    #if model.predict() return [3] so [0] get the first value->3. so prediction=3(which means poor)
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