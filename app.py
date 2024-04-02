from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# Define a Pydantic model for input parameters
class InputParams(BaseModel):
    Gender: int
    Race: int
    RenalDiseaseIndicator: int
    State: int
    County: int
    IPAnnualReimbursementAmt: float
    IPAnnualDeductibleAmt: float
    OPAnnualReimbursementAmt: float
    OPAnnualDeductibleAmt: float
    ClmAdmitDiagnosisCode: int
    DeductibleAmtPaid: float
    DiagnosisGroupCode: int
    ChronicCondScore: int
    Provider_numeric: int
    BeneID_numeric: int
    AttendingPhysician_numeric: int
    OperatingPhysician_numeric: int
    OtherPhysician_numeric: int
    DiagnosisScore: int
    ProcedureScore: int

# Create a FastAPI app instance
app = FastAPI()

# Load the CatBoostRegressor model from pickle dump
with open("catboost_model.pkl", "rb") as file:
    catboost_model = pickle.load(file)


@app.post('/predict')
async def predict(payload : InputParams):
    data = payload.__dict__
    print(data)
    print("Hello")
 
    # Extract values from the dictionary
    features = [data[field] for field in InputParams.model_fields.keys()]
 
    # Make prediction
    prediction = catboost_model.predict([features])[0]
 
    print("Predicted insurance amount is : ", prediction)
 
    return {
        'Prediction' : prediction
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
