import numpy as np
import cv2
import struct
from pathlib import Path

# from https://www.kaggle.com/datasets/hojjatk/mnist-dataset
MNIST_DIR = Path(__file__).resolve().parent.parent.parent / 'data' / 'MNIST'

def get_images(train_file_name = 'train-images.idx3-ubyte', test_file_name = 't10k-images.idx3-ubyte'):
    '''
        Returns np.array of all images in the MNIST dataset with explicit channel dimensions
    '''

    with open(MNIST_DIR/train_file_name, 'rb') as f:      
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))

        raw_data = np.fromfile(f, dtype=np.uint8)
        # Scaled values 0 - 1 so CNN doesnt explode
        train = raw_data.reshape((num, 1, rows, cols)) / 255.0

    with open(MNIST_DIR/test_file_name, 'rb') as f:      
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))

        raw_data = np.fromfile(f, dtype=np.uint8)
        test = raw_data.reshape((num, 1, rows, cols)) / 255.0

    return train, test
    
def get_labels(train_file_name = 'train-labels.idx1-ubyte', test_file_name = 't10k-labels.idx1-ubyte'):
    with open(MNIST_DIR/train_file_name, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        train = np.frombuffer(f.read(), dtype=np.uint8)

    with open(MNIST_DIR/test_file_name, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        test = np.frombuffer(f.read(), dtype=np.uint8)

    return train, test


def display_images(images, titles=None):
    '''
        Displays every image in the parameter array, quit by pressing q
    '''
    for idx, image in enumerate(images):
        img = np.squeeze(image)

        # Bring scale back up to 0-255 integers if it was normalized
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Resize for better visibility
        resized_img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)

        # Use custom window title if provided
        window_title = titles[idx] if titles is not None else "MNIST images"

        # Render image window
        cv2.imshow(window_title, resized_img)

        # Break loop if 'q' is pressed, otherwise show next image every 100ms
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()