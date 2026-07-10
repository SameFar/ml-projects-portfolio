import numpy as np
from neural_network import NeuralNetwork, ol
from data_preprocessing import get_data
from conversion import get_y_dict
from save_model import save_model

EPOCH = 3000

model = NeuralNetwork()
X, y = get_data()
batch_size = X.shape[0]
y_train_dict = get_y_dict(y)

for epoch in range(EPOCH):
    preds = model.forward(X)

    total_epoch_loss = 0.0
    loss_tracking = {}

    for name, meta in ol.items():
        y_true = y_train_dict[name]
        y_pred = preds[name]

        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        if meta[1] == "mc":
            # Categorical Cross Entropy Loss for Softmax outputs
            head_loss = -1 / batch_size * np.sum(y_true * np.log(y_pred))
        else:
            # Binary Cross Entropy Loss for Sigmoid outputs
            # Formula: -1/m * sum(y*log(p) + (1-y)*log(1-p))
            head_loss = (
                -1
                / batch_size
                * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            )

        # Individual head loss
        loss_tracking[name] = head_loss
        # Total error
        total_epoch_loss += head_loss

    model.backward(X, y_train_dict, preds)

    # get starting states and states after every 50 epochs
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"\n================ EPOCH {epoch + 1} ================")
        print(f"Total Aggregated Loss: {total_epoch_loss:.4f}")
        print("-" * 40)
        for name, loss_val in loss_tracking.items():
            print(f" -> {name} Loss: {loss_val:.4f}")

save_model(model)
