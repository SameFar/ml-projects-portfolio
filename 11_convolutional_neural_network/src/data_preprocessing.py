import numpy as np
import cv2
import struct
from pathlib import Path


MNIST_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'MNIST'

with open(MNIST_DIR/'train-images.idx3-ubyte', 'rb') as f:
    magic, num, rows, cols = struct.unpack('>IIII', f.read(16))

    raw_data = np.fromfile(f, dtype=np.uint8)

    images = raw_data.reshape((num, rows, cols))

for i in range(num):
    img = images[i]

    # Resize for better visibility (MNIST images are tiny, e.g., 28x28)
    resized_img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)

    # Render image window
    cv2.imshow("Ubyte Image Viewer", resized_img)

    # Break loop if 'q' is pressed, otherwise show next image every 500ms
    if cv2.waitKey(500) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()