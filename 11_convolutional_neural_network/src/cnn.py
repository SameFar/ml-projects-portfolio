import numpy as np
from scipy import signal

class CompleteCNN:
    def __init__(self, learning_rate=0.01, input_size=(1, 28, 28), kernel_size=3, depth=4, num_classes=10):
        # Convolutional Properties
        self.in_depth, in_height, in_width = input_size
        self.depth = depth
        self.in_shape = input_size
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size

        # Output shape for valid convolution
        self.conv_output_shape = (depth, in_height - kernel_size + 1, in_width - kernel_size + 1)
        self.kernals_shape = (depth, self.in_depth, kernel_size, kernel_size)
        
        # Scaled Weights Initialization
        self.kernals = np.random.randn(*self.kernals_shape) * 0.1
        self.conv_biases = np.random.randn(*self.conv_output_shape) * 0.1

        # Flattened size = depth * height * width
        self.flat_size = depth * self.conv_output_shape[1] * self.conv_output_shape[2]
        self.num_classes = num_classes
        
        # Dense Weights Initialization
        self.dense_weights = np.random.randn(num_classes, self.flat_size) * 0.1
        self.dense_biases = np.random.randn(num_classes, 1) * 0.1

    # Internal Utilities
    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_backward(self, gradient, pre_activation):
        return gradient * (pre_activation > 0)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, X):
        """
        Accepts np.array image, passes it through the entire network, 
        and returns a 10x1 vector of class probabilities.
        """
        self.X = X
        
        # Kernal pass
        self.conv_out = np.copy(self.conv_biases)
        for i in range(self.depth):
            for j in range(self.in_depth):
                self.conv_out[i] += signal.correlate2d(self.X[j], self.kernals[i, j], 'valid')
        
        # relu activation and flattening
        self.activated_out = self._relu(self.conv_out)
        self.flat_out = self.activated_out.reshape(-1, 1)
        
        # Dense pass
        self.dense_out = np.dot(self.dense_weights, self.flat_out) + self.dense_biases
        
        # Output
        self.probabilities = self._softmax(self.dense_out)
        return self.probabilities

    def backward(self, target_label):
        """
        Accepts the true integer class label, computes gradients 
        internally across all layers, and updates all weights.
        """
        # The derivative of Cross-Entropy Loss with Softmax simplified
        loss_gradient = np.copy(self.probabilities)
        loss_gradient[target_label, 0] -= 1.0 

        # Dense layer gradient
        dense_weights_gradient = np.dot(loss_gradient, self.flat_out.T)
        dense_input_gradient = np.dot(self.dense_weights.T, loss_gradient)
        
        # unflatten gradient
        flat_gradient = dense_input_gradient.reshape(self.conv_output_shape)
        
        # relu gradient
        conv_gradient = self._relu_backward(flat_gradient, self.conv_out)
        
        # conv gradient
        kernal_gradient = np.zeros(self.kernals_shape)
        input_gradient = np.zeros(self.in_shape)

        for i in range(self.depth):
            for j in range(self.in_depth):
                kernal_gradient[i, j] = signal.correlate2d(self.X[j], conv_gradient[i], 'valid')
                input_gradient[j] += signal.correlate2d(conv_gradient[i], self.kernals[i, j], 'full')
        
        # Gradient descent
        # Update Dense Parameters
        self.dense_weights -= self.learning_rate * dense_weights_gradient
        self.dense_biases -= self.learning_rate * loss_gradient
        
        # Update Conv Parameters
        self.kernals -= self.learning_rate * kernal_gradient
        self.conv_biases -= self.learning_rate * conv_gradient

        return input_gradient
