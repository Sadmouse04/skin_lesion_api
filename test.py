import requests
import json

# Define the URL of local FastAPI server
url = "http://127.0.0.1:8000/predict"

# JSON payload expected by Pydantic model
# Ensure the img_id actually exist in the local dataset image folder (data/images)
# If want to test other image , please add the specific image into the dataset image folder (data/images)

payload = {
    "age": 68.0,
    "region_ARM": 0,
    "region_NECK": 0,
    "region_FACE": 0,
    "region_HAND": 0,
    "region_FOREARM": 1,
    "region_CHEST": 0,
    "region_NOSE": 0,
    "region_THIGH": 0,
    "region_SCALP": 0,
    "region_EAR": 0,
    "region_BACK": 0,
    "region_FOOT": 0,
    "region_ABDOMEN": 0,
    "region_LIP": 0,
    "itch_False": 0,
    "itch_True": 1,
    "grew_False": 1,
    "grew_True": 0,
    "hurt_False": 1,
    "hurt_True": 0,
    "changed_False": 0,
    "changed_True": 1,
    "bleed_False": 1,
    "bleed_True": 0,
    "elevation_False": 1,
    "elevation_True": 0
}

print("Sending request to API...")
r = requests.post(url, json=payload)

print(json.dumps(r.json(), indent=2))