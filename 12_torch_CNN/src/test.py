import logging
import torch


def test_model(model, test_loader, device):
    model.eval()

    correct_predictions = 0
    total_samples = 0

    # Disable gradient calculation for speed and memory efficiency
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_val_acc = (correct_predictions / total_samples) * 100

    # Comprehensive logging
    logging.info(f"Val Acc: {epoch_val_acc:.2f}%")
