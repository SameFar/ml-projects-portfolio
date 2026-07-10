import numpy as np
from data_preprocessing import get_labels, get_images
from cnn import CompleteCNN
from save_model import save_model

# Load dataset
X_train, X_test = get_images()
y_train, y_test = get_labels()
epochs = 100  # Should take awhile
steps = 0

model = CompleteCNN(learning_rate=0.01)

# Training loop
print("Starting Training...")
for epoch in range(epochs):
    correct_train = 0

    for x_sample, y_label in zip(X_train, y_train):
        predictions = model.forward(x_sample)
        model.backward(y_label)

        if np.argmax(predictions) == y_label:
            correct_train += 1

    train_acc = (correct_train / len(X_train)) * 100
    print(f"Epoch {epoch + 1}/{epochs} Completed | Train Accuracy: {train_acc:.2f}%")

    if train_acc > 99:
        steps += 1
    if train_acc == 100 or steps == 10:
        break

print("\nTraining Finished.")
save_model(model)
