import numpy as np
from save_model import load_model
from PIL import Image
import numpy as np
from pathlib import Path

IMG_DIR = Path(__file__).resolve().parent / 'added_images'
model = load_model()

for files in IMG_DIR.iterdir():
    
    # load image
    file = files.name
    img = Image.open(IMG_DIR / file)

    # transform it into expected input shape (1,28,28)
    img_gray = img.convert('L')
    img_resized = img_gray.resize((28, 28))
    X_raw = np.array(img_resized)
    X = X_raw[np.newaxis, :, :]
    X = X / 255.0

    y = file[0]
    
    preds = model.forward(X, is_training=False)
    predicted_label = np.argmax(preds)

    print(f'Predicted = {predicted_label},  Actual = {y}')