# src/app.py
import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Dynamic absolute path resolution targeting the project directory structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, 'saved_models', 'housing_xgb_pipeline.pkl')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Production model weight artifact not resolved at: {MODEL_PATH}. Run training pipeline first.")


with open(MODEL_PATH, 'rb') as f:
    artifact = pickle.load(f)

model_pipeline = artifact['pipeline']
expected_features = artifact['features']

# Price category mapper definition matching training discretization labels
PRICE_BRACKETS = {
    0: "Under 5 Crore PKR",
    1: "5 to 10 Crore PKR",
    2: "10 to 15 Crore PKR",
    3: "15 to 20 Crore PKR",
    4: "Above 20 Crore PKR"
}

app = FastAPI(
    title="Islamabad Housing Market Prediction Engine",
    description="Production XGBoost classification API utilizing macro-stratified spatial features.",
    version="1.0.0"
)

class PropertyInferenceSchema(BaseModel):
    baths: int = Field(..., description="Number of bathrooms", json_schema_extra={"example": 3})
    beds: int = Field(..., description="Number of bedrooms", json_schema_extra={"example": 4})
    marla: float = Field(..., description="Total land area computed in Marlas", json_schema_extra={"example": 10.0})
    rooms: int = Field(..., description="Count of extra premium rooms (Drawing/Powder/Steam/etc.)", json_schema_extra={"example": 1})
    location: str = Field(..., description="Specific sector or neighborhood name", json_schema_extra={"example": "G-13"})

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_market_bracket(payload: PropertyInferenceSchema):
    try:
        # Construct template base input vector matching training schemas
        input_data = {feat: [0] for feat in expected_features}
        
        # Hydrate primary numeric features
        input_data['Baths'] = [payload.baths]
        input_data['Beds'] = [payload.beds]
        input_data['Marla'] = [payload.marla]
        input_data['Other Rooms'] = [payload.rooms]
        
        # Set target one-hot field if it wasn't dropped during baseline training
        target_location = payload.location.strip()
        if target_location in input_data:
            input_data[target_location] = [1]
            
        # Standardize structured matrix dataframe matching training data column sequence exactly
        input_df = pd.DataFrame(input_data)[expected_features]
        
        # Execute live prediction pipeline processing
        class_idx = int(model_pipeline.predict(input_df)[0])
        price_range = PRICE_BRACKETS.get(class_idx, "Unknown Valuation Spectrum")
        
        return {
            "prediction_index": class_idx,
            "predicted_price_bracket": price_range
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Runtime Fault: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)