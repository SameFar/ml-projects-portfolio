import pandas as pd
import torch
from pathlib import Path
from .stock_lstm import StockLSTM

def predict_tomorrow_open():
    print("\n=== STARTING LOCAL CSV PREDICTION PIPELINE ===")
    
    csv_path = Path(__file__).resolve().parent.parent / 'nvidea_stocks2.csv'
    print(f"Reading normalized dataset from: {csv_path}")
    
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    
    # Prediction date
    target_date = '2026-04-10'
    if target_date not in df.index:
        target_date = pd.to_datetime(target_date)
        if target_date not in df.index:
            raise KeyError(f"Target date {target_date} not found in the dataset.")
            
    target_idx = df.index.get_loc(target_date)
    
    start_idx = target_idx - 50
    if start_idx < 0:
        raise ValueError(f"Not enough historical rows before {target_date} to build a 50-day window.")
        
    recent_history = df.iloc[start_idx:target_idx, 0:3].values
    print(f"Extracted feature window shape: {recent_history.shape}")
    
    X_input = torch.tensor(recent_history, dtype=torch.float32).unsqueeze(0)
    
    model_path = Path(__file__).resolve().parent.parent / 'model' / 'model.pt'
    model = StockLSTM(input_size=3, hidden_size=50, num_layers=1)
    
    print("Loading model state weights...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        scaled_pred = model(X_input).item()
        
    MIN = 10.943877
    MAX = 207.814954
    predicted_open_usd = (scaled_pred * (MAX - MIN)) + MIN
    
    print(f"\nPredicted NVDA Opening Price for {target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else target_date}: ${predicted_open_usd:.2f}")
    print("=== PIPELINE FINISHED SUCCESSFULLY ===")
    return predicted_open_usd
