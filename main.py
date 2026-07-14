import torch
import pytorch_lightning as pl

import json
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from src.utils.utils import dict_to_namespace
from src.late_fusion.late_fusion import late_fusion_predict_prob

app = FastAPI()

# Global variable to hold your config which can access to the trained model
config = None

# Load the configuration
@app.on_event("startup")
def load_config():
    global config
    config_path = "config.json"
    with open(config_path, "r") as f:
        raw_config = json.load(f)
        
    # Replicate Autoprognosis-M's configuration merging logic for late_fusion
    globals_cfg = raw_config["globals"]
    
    # Extract the late_fusion configuration (Index 2 based on config.json)
    lf_cfg = raw_config["configurations"][2] 

    # Merge globals with specific module configs and convert to SimpleNamespace
    config = dict_to_namespace({**globals_cfg, **lf_cfg})
    config.imaging = dict_to_namespace({**globals_cfg, **lf_cfg["imaging"]})
    config.tabular = dict_to_namespace({**globals_cfg, **lf_cfg["tabular"]})
    
    print("Late Fusion Configuration loaded successfully.")

class Patient(BaseModel):
    img_id: str
    age: float
    region_ARM: int
    region_NECK: int
    region_FACE: int
    region_HAND: int
    region_FOREARM: int
    region_CHEST: int
    region_NOSE: int
    region_THIGH: int
    region_SCALP: int
    region_EAR: int
    region_BACK: int
    region_FOOT: int
    region_ABDOMEN: int
    region_LIP: int
    itch_False: int
    itch_True: int
    grew_False: int
    grew_True: int
    hurt_False: int
    hurt_True: int
    changed_False: int
    changed_True: int
    bleed_False: int
    bleed_True: int
    elevation_False: int
    elevation_True: int
    
    # IMPORTANT: Dummy target column to prevent Pandas KeyError during tabular prediction
    # because the API is reusing the code that was originially written for training
    # it can still do prediction even telling model that diagnostic is "NEV"
    diagnostic: str = "NEV"

@app.post("/predict")
def predict(data: Patient):
    # Convert the incoming Pydantic data into a dictionary
    input_dict = data.model_dump()
    
    # Transform into a 1-row Pandas DataFrame
    df = pd.DataFrame([input_dict])
    
    # Pass the DataFrame and config to your existing function
    probabilities_df = late_fusion_predict_prob(config, df=df)
    raw_probs = probabilities_df.iloc[0]
        
    # Extract the final prediction (6-way classification)
    predicted_class = raw_probs.idxmax()
    max_probability = raw_probs.max()

    # binary classification (Malignant vs Benign)
    malignant_classes = ["BCC", "MEL", "SCC"]
    benign_classes = ["ACK", "NEV", "SEK"]

    malignant_prob = raw_probs[malignant_classes].sum()
    benign_prob = raw_probs[benign_classes].sum()

    binary_prediction = "Malignant" if malignant_prob > benign_prob else "Benign"
    binary_confidence = max(malignant_prob, benign_prob)

    return{
        "binary_prediction": binary_prediction,
        "binary_confidence": float(binary_confidence),
        "multi_class_prediction": predicted_class,
        "multi_class_confidence": float(max_probability),
        "raw_probabilities": raw_probs.to_dict()
    }