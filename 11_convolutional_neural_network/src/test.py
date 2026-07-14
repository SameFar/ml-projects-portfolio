import numpy as np
from .data_preprocessing import get_labels, get_images, display_images
from .save_model import load_model

# Load dataset
X_train, X_test = get_images()
y_train, y_test = get_labels()

model = load_model()


# Testing
correct_test = 0
failed_images = []
failed_true_labels = []
failed_predicted_labels = []

for x_sample, y_label in zip(X_test, y_test):
    predictions = model.forward(x_sample, is_training=False)
    predicted_label = np.argmax(predictions)

    if predicted_label == y_label:
        correct_test += 1
    else:
        # If the network failed, log the image and the mix-up details
        failed_images.append(x_sample)
        failed_true_labels.append(y_label)
        failed_predicted_labels.append(predicted_label)

# Calculate final test accuracy
test_acc = (correct_test / len(X_test)) * 100
print(f"Final Test Accuracy: {test_acc:.2f}%")
print(f"Total Misclassifications: {len(failed_images)} out of {len(X_test)}")


if len(failed_images) > 0:
    print("\nDisplaying failed predictions...")

    failed_images_arr = np.array(failed_images)

    # To set display title
    custom_titles = [
        f"True: {t} | Pred: {p}"
        for t, p in zip(failed_true_labels, failed_predicted_labels)
    ]

    display_images(failed_images_arr, custom_titles)
else:
    print("\nIncredible! The model didn't fail a single test.")
