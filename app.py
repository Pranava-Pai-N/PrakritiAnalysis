from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fpdf import FPDF
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
# Load models
model = joblib.load("prakrithi_xgboost_model.pkl")
encoder = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define Request Body Schema
class PrakrithiRequest(BaseModel):
    Name: str
    Age: int
    Gender: str
    Height: float
    Weight: float
    Body_Type: str
    Skin_Type: str
    Hair_Type: str
    Facial_Structure: str
    Complexion: str
    Eyes: str
    Food_Preference: str
    Bowel_Movement: str
    Thirst_Level: str
    Sleep_Duration: float
    Sleep_Quality: str
    Energy_Levels: str
    Daily_Activity_Level: str
    Exercise_Routine: str
    Food_Habit: str
    Water_Intake: str
    Health_Issues: str
    Hormonal_Imbalance: str
    Skin_Hair_Problems: str
    Ayurvedic_Treatment: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Prakrithi Analysis API!"}

@app.post("/generate_pdf/")
def generate_pdf(data: PrakrithiRequest):
    try:
        df = pd.DataFrame([data.dict()])

        categorical_cols = [
            'Gender', 'Body_Type', 'Skin_Type', 'Hair_Type', 'Facial_Structure', 'Complexion',
            'Eyes', 'Food_Preference', 'Bowel_Movement', 'Thirst_Level', 'Sleep_Quality',
            'Energy_Levels', 'Daily_Activity_Level', 'Exercise_Routine', 'Food_Habit',
            'Water_Intake', 'Health_Issues', 'Hormonal_Imbalance', 'Skin_Hair_Problems',
            'Ayurvedic_Treatment'
        ]
        
        X_categorical = encoder.transform(df[categorical_cols])
        X_categorical = pd.DataFrame(X_categorical, columns=encoder.get_feature_names_out())

        X_numeric = df[['Age', 'Height', 'Weight', 'Sleep_Duration']]

        X_combined = pd.concat([X_numeric, X_categorical], axis=1)

        X_scaled = scaler.transform(X_combined)

        prediction = model.predict(X_scaled)
        prakrithi = label_encoder.inverse_transform(prediction)[0]

        response = {
            "Name": data.Name,
            "Age": data.Age,
            "Gender": data.Gender,
            "Dominant_Prakrithi": {
                "text":prakrithi,
                "style":"bold"
            },
            "Body_Constituents": {
                "Body_Type": data.Body_Type,
                "Skin_Type": data.Skin_Type,
                "Hair_Type": data.Hair_Type,
                "Thirst_Level": data.Thirst_Level,
                "Sleep_Pattern": data.Sleep_Quality,
            },
            "Potential_Health_Concerns": [
                "Acidity & Gastric Issues",
                "Skin Rashes & Acne",
                "Liver Disorders",
                "Obesity & Joint Pain"
            ],
            "Recommendations": {
                "Dietary_Guidelines": [
                    "Avoid spicy, oily foods",
                    "Eat cooling and light foods like cucumber, watermelon",
                    "Reduce dairy intake"
                ],
                "Lifestyle_Suggestions": [
                    "Regular meditation & stress management",
                    "Moderate exercise like yoga"
                ],
                "Ayurvedic_Herbs_Remedies": {
                    "For_Digestion": ["Triphala", "Aloe Vera Juice"],
                    "For_Skin": ["Neem", "Turmeric"]
                }
            }
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
