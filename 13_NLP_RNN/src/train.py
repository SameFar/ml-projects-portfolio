import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .data_processing import train_test_split
import logging

def train(rnn, data, n_epoch = 45, n_batch_size = 64, report_every = 5, learning_rate = 0.2, criterion = nn.NLLLoss(), export_path="char_rnn.pt"):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    current_loss = 0
    all_losses = []
    
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    logging.debug('Starting training')
    logging.info(f'')

    training_data, val_data = train_test_split(data)

    train_loader = DataLoader(
        training_data, 
        batch_size=n_batch_size, 
        shuffle=True, 
        collate_fn=lambda x: x
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=n_batch_size, 
        shuffle=False,
        collate_fn=lambda x: x
    )

    for epoch in range(1, n_epoch + 1):
        logging.debug(epoch)
        rnn.train() # Ensure model is in training mode
        rnn.zero_grad() 

        for batch in train_loader:
            batch_loss = 0
            
            for (label_tensor, text_tensor, label, text) in batch: 
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(train_loader))
        
        rnn.eval()
        correct = 0
        total = 0
        
        with torch.no_grad(): # Disables gradient calculation to save memory and compute
            for batch in val_loader:
                for (label_tensor, text_tensor, label, text) in batch:
                    output = rnn.forward(text_tensor)
                    top_n, top_i = output.topk(1)
                    pred_label_idx = top_i[0].item()
                    true_label_idx = label_tensor.item()
                    
                    if pred_label_idx == true_label_idx:
                        correct += 1
                    total += 1
        
        val_accuracy = correct / total if total > 0 else 0

        if epoch % report_every == 0:
            logging.info(f"{epoch} ({epoch / n_epoch:.0%}): \t average batch loss = {all_losses[-1]:.4f} \t Val Accuracy = {val_accuracy:.2%}")
        current_loss = 0

    logging.debug(f"Training complete.")
    torch.save(rnn.state_dict(), export_path)

    return all_losses