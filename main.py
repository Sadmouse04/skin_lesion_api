import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.pkl") 

feature_columns = [
    "age", 
    "region_ABDOMEN", "region_ARM", 
    "region_BACK", "region_CHEST", 
    "region_EAR", "region_FACE", 
    "region_FOOT", "region_FOREARM", 
    "region_HAND", "region_LIP", 
    "region_NECK", "region_NOSE", 
    "region_SCALP", "region_THIGH",
    "itch_False", "itch_True",
    "grew_False", "grew_True",
    "hurt_False", "hurt_True",
    "changed_False", "changed_True",
    "bleed_False", "bleed_True",
    "elevation_False", "elevation_True"
]

class Patient(BaseModel):
    age: float
    region_ABDOMEN: int = 0
    region_ARM: int = 0
    region_BACK: int = 0
    region_CHEST: int = 0
    region_EAR: int = 0
    region_FACE: int = 0
    region_FOOT: int = 0
    region_FOREARM: int = 0
    region_HAND: int = 0
    region_LIP: int = 0
    region_NECK: int = 0
    region_NOSE: int = 0
    region_SCALP: int = 0
    region_THIGH: int = 0
    itch_False: int = 0
    itch_True: int = 0
    grew_False: int = 0
    grew_True: int = 0
    hurt_False: int = 0
    hurt_True: int = 0
    changed_False: int = 0
    changed_True: int = 0
    bleed_False: int = 0
    bleed_True: int = 0
    elevation_False: int = 0
    elevation_True: int = 0
    
@app.post("/predict")
def predict(data: Patient):
    input_dict = data.model_dump()

    df = pd.DataFrame([input_dict])
    df = df[feature_columns]
    
    # probabilities for 6 classes
    probabilities = model.predict_proba(df)[0]
    class_names = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
    prob_dict = dict(zip(class_names, probabilities))

    # Multi-class
    predicted_class = max(prob_dict, key=prob_dict.get)
    max_probability = prob_dict[predicted_class]

    # Binary-class
    malignant_classes = ["BCC", "MEL", "SCC"]
    benign_classes = ["ACK", "NEV", "SEK"]

    malignant_prob = sum(prob_dict[cls] for cls in malignant_classes)
    benign_prob = sum(prob_dict[cls] for cls in benign_classes)

    binary_prediction = "Malignant" if malignant_prob > benign_prob else "Benign"
    binary_confidence = max(malignant_prob, benign_prob)

    return {
        "binary_prediction": binary_prediction,
        "binary_confidence": float(binary_confidence),
        "multi_class_prediction": predicted_class,
        "multi_class_confidence": float(max_probability),
        "raw_probabilities": {k: float(v) for k, v in prob_dict.items()}
    }