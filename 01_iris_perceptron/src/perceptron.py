import numpy as np

class Perceptron:
    def __init__(self, inputs=4, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = np.zeros(inputs)
        self.bias = 0.0

    def activate(self, X: np.ndarray) -> int:
        guess = np.dot(self.weights, X) + self.bias
        return 1 if guess > 0 else 0
    
    def train(self, X: np.ndarray, y: int) -> int:
        guess = self.activate(X)
        error = y - guess
        
        # Weight adjustments
        self.weights += self.lr * error * X
        self.bias += self.lr * error
        
        return error