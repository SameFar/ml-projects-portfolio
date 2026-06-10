from fastapi import FastAPI
from fastapi.responses import JSONResponse
import torch
from pydantic_model import FashionInput
from conversion import convert, y_convert
from neural_network import MultiHeadNeuralNetwork, ol
from pathlib import Path

model_path = Path(__file__).resolve().parent / 'model' / "torch_model.pth"

app = FastAPI()

def load_pytorch_model(model_path=model_path):
    """
    Instantiates the model structure and loads the saved weights.
    """
    model = MultiHeadNeuralNetwork(input_size=20, outputs=ol)
    
    # Load the saved weights dictionary into the structure
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

model = load_pytorch_model(model_path)

@app.get('/')
async def home():
    return {'health_check': 'running'}

@app.post('/predict')
async def predict(values: FashionInput):
    input_array = convert(values).reshape(1, -1)
    
    # Convert NumPy array to a Float32 PyTorch Tensor
    model_input = torch.tensor(input_array, dtype=torch.float32)
    
    # Inference without tracking gradients
    with torch.no_grad():
        pred_tensors = model(model_input)
        
    # Convert PyTorch tensor outputs back to NumPy arrays for y_convert function
    # .detach() removes it from any graph, .cpu() ensures it's on RAM, .numpy() converts it
    pred = {
        name: tensor.detach().cpu().numpy() 
        for name, tensor in pred_tensors.items()
    }
    
    # Map probabilities/values back to strings/readable labels
    outs = y_convert(pred)
    return JSONResponse(outs)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)