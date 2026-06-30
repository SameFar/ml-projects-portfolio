import pandas as pd
import torch
import numpy as np
from pathlib import Path
from .stock_lstm import StockLSTM

def predict_tomorrow_open(target_date = '2026-05-20'):
    
    csv_path = Path(__file__).resolve().parent.parent / 'nvidea_stocks_pred.csv'    
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    
    target_date = pd.to_datetime(target_date)
    
    if target_date not in df.index:
        raise KeyError(f"Target date {target_date.strftime('%Y-%m-%d')} not found in the dataset.")
            
    target_idx = df.index.get_loc(target_date)
    
    start_idx = target_idx - 50
    if start_idx < 0:
        raise ValueError(f"Not enough historical rows before {target_date} to build a 50-day window.")
        
    # Grabs the 50 days leading up to and including target_date
    recent_history = df.iloc[start_idx:target_idx + 1, 0:3].values
    
    X_input = torch.tensor(recent_history, dtype=torch.float32).unsqueeze(0)
    
    model_path = Path(__file__).resolve().parent.parent / 'model' / 'model.pt'
    model = StockLSTM(input_size=3, hidden_size=50, num_layers=1)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        predicted_log_return = model(X_input).item()
    
    # reverse scaling
    today_actual_close = df.iloc[target_idx]['Close']
    predicted_open_usd = today_actual_close * np.exp(predicted_log_return)
    
    print(f"\nPredicted NVDA Opening Price for tomorrow: ${predicted_open_usd:.2f}")
    return predicted_open_usd