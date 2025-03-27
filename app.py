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
        # Convert request data to DataFrame
        df = pd.DataFrame([data.dict()])

        # Categorical Features
        categorical_cols = [
            'Gender', 'Body_Type', 'Skin_Type', 'Hair_Type', 'Facial_Structure', 'Complexion',
            'Eyes', 'Food_Preference', 'Bowel_Movement', 'Thirst_Level', 'Sleep_Quality',
            'Energy_Levels', 'Daily_Activity_Level', 'Exercise_Routine', 'Food_Habit',
            'Water_Intake', 'Health_Issues', 'Hormonal_Imbalance', 'Skin_Hair_Problems',
            'Ayurvedic_Treatment'
        ]
        
        # Encode categorical data
        X_categorical = encoder.transform(df[categorical_cols])
        X_categorical = pd.DataFrame(X_categorical, columns=encoder.get_feature_names_out())

        # Extract numerical features
        X_numeric = df[['Age', 'Height', 'Weight', 'Sleep_Duration']]

        # Combine numerical and categorical data
        X_combined = pd.concat([X_numeric, X_categorical], axis=1)

        # Scale features
        X_scaled = scaler.transform(X_combined)

        # Make prediction
        prediction = model.predict(X_scaled)
        prakrithi = label_encoder.inverse_transform(prediction)[0]

        # Create PDF Report
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Prakrithi Analysis Report", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Patient Name: {data.Name}", ln=True)
        pdf.cell(200, 10, f"Age: {data.Age}", ln=True)
        pdf.cell(200, 10, f"Gender: {data.Gender}", ln=True)
        pdf.cell(200, 10, f"Dominant Prakrithi: {prakrithi}", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Body Constituents & Characteristics", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Body Type: {data.Body_Type}", ln=True)
        pdf.cell(200, 10, f"Skin Type: {data.Skin_Type}", ln=True)
        pdf.cell(200, 10, f"Hair Type: {data.Hair_Type}", ln=True)
        pdf.cell(200, 10, f"Thirst Level: {data.Thirst_Level}", ln=True)
        pdf.cell(200, 10, f"Sleep Pattern: {data.Sleep_Quality}", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Potential Health Concerns", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Potential concerns based on Prakrithi:", ln=True)
        pdf.cell(200, 10, "Acidity & Gastric Issues", ln=True)
        pdf.cell(200, 10, "Skin Rashes & Acne", ln=True)
        pdf.cell(200, 10, "Liver Disorders", ln=True)
        pdf.cell(200, 10, "Obesity & Joint Pain", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Recommendations", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Dietary Guidelines:", ln=True)
        pdf.cell(200, 10, "Avoid spicy, oily foods", ln=True)
        pdf.cell(200, 10, "Eat cooling and light foods like cucumber, watermelon", ln=True)
        pdf.cell(200, 10, "Reduce dairy intake", ln=True)
        pdf.ln(5)

        pdf.cell(200, 10, "Lifestyle Suggestions:", ln=True)
        pdf.cell(200, 10, "Regular meditation & stress management", ln=True)
        pdf.cell(200, 10, "Moderate exercise like yoga", ln=True)
        pdf.ln(5)

        pdf.cell(200, 10, "Ayurvedic Herbs & Remedies:", ln=True)
        pdf.cell(200, 10, "For Digestion: Triphala, Aloe Vera Juice", ln=True)
        pdf.cell(200, 10, "For Skin: Neem, Turmeric", ln=True)

        pdf_path = "prakrithi_report.pdf"
        pdf.output(pdf_path, "F")

        return FileResponse(pdf_path, media_type="application/pdf", filename=f"{data.Name}.pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
