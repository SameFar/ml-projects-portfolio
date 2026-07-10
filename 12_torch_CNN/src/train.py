import torch
import torch.nn as nn
import torch.optim as optim
import logging


def train_model(model, train_loader, val_loader, device, epochs=10):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        logging.debug(f"{epoch + 1}/{epochs} | Training Done ")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Disable gradient calculation for speed and memory efficiency
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        epoch_val_loss = val_loss / total_samples
        epoch_val_acc = (correct_predictions / total_samples) * 100

        # Comprehensive logging
        logging.info(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.2f}%"
        )

    return model
