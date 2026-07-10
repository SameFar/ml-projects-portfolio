import torch
import torch.nn.functional as F
from neural_network import MultiHeadNeuralNetwork, ol
from data_preprocessing import get_data
from conversion import get_y_dict
from pathlib import Path

model_path = Path(__file__).resolve().parent / "model" / "torch_model.pth"
EPOCH = 2000

# Initialize the PyTorch Model and Optimizer
model = MultiHeadNeuralNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Get Data and convert to PyTorch Tensors
X_np, y_np = get_data()
X = torch.tensor(X_np, dtype=torch.float32)

y_train_dict_np = get_y_dict(y_np)
# Convert all target dictionary arrays into float32 tensors
y_train_dict = {
    name: torch.tensor(arr, dtype=torch.float32)
    for name, arr in y_train_dict_np.items()
}

# Training Loop
for epoch in range(EPOCH):
    # Zero out the gradients from the previous step
    optimizer.zero_grad()

    # Forward pass
    preds = model(X)

    total_epoch_loss = 0.0
    loss_tracking = {}

    for name, meta in ol.items():
        y_true = y_train_dict[name]
        y_pred = preds[name]

        if meta[1] == "mc":
            # PyTorch expects probabilities for kl_div or manual log-sum,
            # but since you already applied softmax in forward(), we do it like this:
            head_loss = -torch.mean(
                torch.sum(y_true * torch.log(torch.clamp(y_pred, 1e-15, 1.0)), dim=-1)
            )
        else:
            # Binary Cross Entropy for Sigmoid outputs
            head_loss = F.binary_cross_entropy(y_pred, y_true, reduction="mean")

        # Track individual head loss (item() converts single-value tensor to float)
        loss_tracking[name] = head_loss.item()

        # Accumulate to the total loss tensor graph
        total_epoch_loss += head_loss

    # Backward pass: Computes gradients across the entire shared network
    total_epoch_loss.backward()

    # Gradient Descent: Updates all weights and biases
    optimizer.step()

    # Handle your manual learning rate decay rule via the optimizer
    for g in optimizer.param_groups:
        if g["lr"] > 0.001:
            g["lr"] -= 0.00001

    # Print metrics every 50 epochs
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"\n================ EPOCH {epoch + 1} ================")
        print(f"Total Aggregated Loss: {total_epoch_loss.item():.4f}")
        print("-" * 40)
        for name, loss_val in loss_tracking.items():
            print(f" -> {name} Loss: {loss_val:.4f}")


torch.save(model.state_dict(), model_path)
