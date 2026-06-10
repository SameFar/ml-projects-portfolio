from fastapi import FastAPI
from fastapi.responses import JSONResponse
from save_model import load_model
from pydantic_model import FashionInput
from conversion import convert, y_convert

app = FastAPI()

model = load_model()

@app.get('/')
async def home():
    return {'health_check' : 'running'}

@app.post('/predict')
async def predict(values : FashionInput):
    input_array = convert(values)
    model_input = input_array.reshape(1, -1)
    
    pred = model.forward(model_input)
    outs = y_convert(pred)
    return JSONResponse(outs)