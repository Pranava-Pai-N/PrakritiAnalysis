from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

app = FastAPI()
load_dotenv() 

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
    ApiKey : str
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

@app.post("/generate_pdf")
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
                "Dominant_Prakrithi": prakrithi,
                "Body_Constituents": {
                    "Body_Type": data.Body_Type,
                    "Skin_Type": data.Skin_Type,
                    "Hair_Type": data.Hair_Type,
                    "Thirst_Level": data.Thirst_Level,
                    "Sleep_Pattern": data.Sleep_Quality,
                },
            }
        if data.ApiKey ==os.getenv("PREMIUM_API_KEY"):
            if prakrithi == "Vata":
                        response["Recommendations"] = {
                            "Dietary_Guidelines": [
                                "Eat warm, cooked foods like soups and porridges.",
                                "Include healthy fats (ghee, sesame oil).",
                                "Prefer sweet, grounding foods (root vegetables, bananas)."
                            ],
                            "Lifestyle_Suggestions": [
                                "Maintain a consistent daily routine (fixed sleep & meals).",
                                "Practice gentle exercises like yoga and walking.",
                                "Apply warm oil massage (Abhyanga) with sesame oil."
                            ],
                            "Ayurvedic_Herbs_Remedies": [
                                "Triphala – Supports digestion.",
                                "Ashwagandha – Reduces stress.",
                                "Aloe Vera & Sesame Oil – Nourish skin and hair."
                            ]
                        }
                        response["Potential_Health_Concerns"]=[
                            "Digestive issues – Gas, bloating, constipation",
                            "Joint problems – Dryness, stiffness, arthritis",
                            "Anxiety & stress – Nervousness, sleep disturbances"
                        ]
                        
            elif prakrithi == "Pitta":
                        response["Recommendations"] = {
                            "Dietary_Guidelines": [
                                "Eat cooling foods like cucumber, coconut water, and leafy greens.",
                                "Avoid spicy, oily, and acidic foods (chili, fried foods, excess citrus).",
                                "Stay hydrated with herbal teas (rose, fennel, coriander)."
                            ],
                            "Lifestyle_Suggestions": [
                                "Maintain cool environments and avoid excessive heat.",
                                "Practice calming exercises like yoga and meditation.",
                                "Engage in stress-relieving activities like deep breathing and nature walks."
                            ],
                            "Ayurvedic_Herbs_Remedies": [
                                "Amla – Supports digestion and reduces acidity.",
                                "Aloe Vera – Cools and soothes skin.",
                                "Rose Water – Refreshes and balances Pitta heat."
                            ]
                        }
                        
                        response["Potential_Health_Concerns"]=[
                            "Acidity & ulcers – Heartburn, acid reflux",
                            "Skin issues – Rashes, acne, inflammation",
                            "Anger & irritability – Prone to stress and mood swings"
                        ]
                        
                        
            elif prakrithi == "Kapha":
                        response["Recommendations"] = {
                            "Dietary_Guidelines": [
                                "Eat light, warm foods like barley, millet, and steamed vegetables.",
                                "Avoid heavy, oily, and sweet foods (fried foods, dairy, excess sugar).",
                                "Include spices like ginger, black pepper, and turmeric to boost metabolism."
                            ],
                            "Lifestyle_Suggestions": [
                                "Engage in regular physical activity (brisk walking, cardio, strength training).",
                                "Avoid daytime naps to prevent sluggishness.",
                                "Stay mentally active with stimulating activities and social engagement."
                            ],
                            "Ayurvedic_Herbs_Remedies": [
                                "Ginger – Improves digestion and boosts metabolism.",
                                "Black Pepper – Reduces mucus buildup and enhances circulation.",
                                "Turmeric – Supports immunity and reduces inflammation."
                            ]
                        }
                        
                        response["Potential_Health_Concerns"]=[
                            "Weight gain & sluggish metabolism – Slow digestion, obesity",
                            "Respiratory issues – Congestion, mucus buildup, sinus problems",
                            "Lethargy & depression – Lack of motivation, drowsiness"
                        ]
                    
            
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
