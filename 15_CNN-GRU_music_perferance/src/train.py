import logging
import torch
import torch.optim as optim
from .save_model import save_model

def train(model, dataloader, epochs=20, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training context initialized on target device: {device}")
    model.to(device)
    
    # Only allocate Adam momentum buffers for parameters requiring gradients (GRU & Head)
    # Added weight_decay (L2 regularization) as a guard against Deep SVDD hypersphere collapse
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)
    
    model.train()

    # Calculate Center
    logging.info("Calculating initial one-class target center...")
    with torch.no_grad():
        initial_embeddings = []
        for batch in dataloader:
            waveforms = batch[0].to(device)  # Route inputs to device
            final_emb = model(waveforms)
            initial_embeddings.append(final_emb)
        
        # Define the fixed centroid point in space from the untuned model
        center = torch.cat(initial_embeddings, dim=0).mean(dim=0, keepdim=True)
        center = center.clone().detach().to(device)

    # Training loop
    for epoch in range(epochs):
        training_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()

            waveforms = batch[0].to(device)  # Route inputs to device
            final_emb = model(waveforms)

            # Deep SVDD Loss: Minimize mean squared Euclidean distance to the center point
            loss = torch.mean(torch.sum((final_emb - center) ** 2, dim=1))
            
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item() * waveforms.size(0)
            
        epoch_loss = training_loss / len(dataloader.dataset)
        logging.info(f"Epoch [{epoch+1}/{epochs}] - Deep SVDD Loss: {epoch_loss:.6f}")
        
    logging.info("One-class model training completed!")
    
    # Check for Hypersphere Collapse
    # Calculate variance across the dataset to ensure the model learned meaningful structures
    with torch.no_grad():
        final_embeddings = []
        for batch in dataloader:
            waveforms = batch[0].to(device)
            final_embeddings.append(model(waveforms))
        all_embs = torch.cat(final_embeddings, dim=0)
        emb_std = torch.std(all_embs, dim=0).mean().item()
        
        logging.info(f"Post-training embedding average standard deviation: {emb_std:.6f}")
        if emb_std < 1e-4:
            logging.warning("⚠️ CRITICAL WARNING: Standard deviation is near zero! The model has likely suffered Hypersphere Collapse.")
            
        # --- DYNAMIC THRESHOLD CALCULATION ---
        # 1. Calculate all squared distances for the training set
        all_distances = []
        for batch in dataloader:
            waveforms = batch[0].to(device)
            final_emb = model(waveforms)
            dist = torch.sum((final_emb - center) ** 2, dim=1)
            all_distances.append(dist)
            
        all_distances = torch.cat(all_distances, dim=0)
        
        # 2. Set threshold at the 90th percentile (90% of liked tracks fall inside)
        all_distances_sorted, _ = torch.sort(all_distances)
        percentile_idx = int(len(all_distances_sorted) * 0.90)
        dynamic_threshold = all_distances_sorted[percentile_idx].item()
        
        logging.info(f"Derived 90th percentile prediction threshold cutoff: {dynamic_threshold:.6f}")

    # Attach properties and save (moving tensors to CPU for storage safety)
    model.center = center.cpu()
    model.threshold = dynamic_threshold
    save_model(model)