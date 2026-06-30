import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path

def train(train_loader, eval_loader, model):

    EPOCH = 1000
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    save_dir = Path(__file__).resolve().parent.parent / 'model'
    save_path = save_dir / 'model.pt'

    for e in range(EPOCH):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for X, y in train_loader:
            optimizer.zero_grad()
            
            pred = model(X)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # Evaluation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X, y in eval_loader:
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(eval_loader)
                    
        print(f'Epoch [{e+1}/{EPOCH}] -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')

    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"\nModel state dictionary successfully saved to: {save_path}")