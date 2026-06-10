from pydantic_model import CustomerYearlyMetrics
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / 'model' 

# Sample labels based on my observations
labels = {
    0:'No Refunds',
    1:'Significant Refund rate',
    2:'Frequent Buyers',
    3:'Anomoly',
    4:'Big Spenders- Infrequent',
    5:'Anomoly'
}

app = FastAPI(
    title="Customer Types",
    version="1.0.0"
)

with open(BASE_DIR /'model.pkl','rb') as f:
    model = pickle.load(f)

@app.get("/")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post('/predict')
async def prediction(cmr : CustomerYearlyMetrics):
    cmr_metrics_dict = {
    "Transactions": cmr.Transactions,
    "Total Quantity": cmr.Quantity,
    "Total Spent": cmr.Spent,
    "Total Refund Recived": cmr.Refunds,
    "Avg Spent": cmr.Avg_Spent,
    "Purchases per month": cmr.Transactions_per_month,
    "Return Rate": cmr.Return_Rate
    }
    pred = model.predict(pd.DataFrame([cmr_metrics_dict]))
    
    prediction_index = int(pred[0])
    predicted_label = labels.get(prediction_index, f"Unknown key received from model: {prediction_index}")

    return JSONResponse(status_code=200, content={'predicted_customer': predicted_label})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)