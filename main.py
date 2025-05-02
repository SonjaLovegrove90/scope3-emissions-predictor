#!/usr/bin/env python
# coding: utf-8

# In[2]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import xgboost as xgb

app = FastAPI()

# Load model and encoder
model = joblib.load("xgb_model.joblib")

# Define input schema
class EmissionsInput(BaseModel):
    commodity: str
    production_value: float
    production_unit: str
    parent_entity: str
    reporting_entity: str
    year: int

@app.post("/predict")
def predict(input_data: EmissionsInput):
    df = pd.DataFrame([input_data.dict()])
    # Preprocess the input here (label encoding, scaling if needed)
    # Apply same transformations used during training

    prediction = model.predict(df)[0]
    return {"predicted_emissions": prediction}



# In[ ]:




