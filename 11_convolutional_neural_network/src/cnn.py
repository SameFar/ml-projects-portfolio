import numpy as np
from scipy import signal

class CompleteCNN:
    def __init__(self, learning_rate=0.01, input_size=(1, 28, 28), kernel_size=3, depth=4, num_classes=10, pool_size=2):
        self.in_depth, in_height, in_width = input_size
        self.depth = depth
        self.in_shape = input_size
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.pool_size = pool_size 
        self.dropout_rate = 0.2

        # Expected shapes
        self.conv_output_shape = (depth, in_height - kernel_size + 1, in_width - kernel_size + 1)
        self.kernals_shape = (depth, self.in_depth, kernel_size, kernel_size)
        
        self.pool_output_shape = (
            depth, 
            self.conv_output_shape[1] // pool_size, 
            self.conv_output_shape[2] // pool_size
        )
        
        # Random (and scaled) weights
        self.kernals = np.random.randn(*self.kernals_shape) * 0.1
        self.conv_biases = np.random.randn(*self.conv_output_shape) * 0.1
        self.flat_size = depth * self.pool_output_shape[1] * self.pool_output_shape[2]
        
        self.dense_weights = np.random.randn(num_classes, self.flat_size) * 0.1
        self.dense_biases = np.random.randn(num_classes, 1) * 0.1

    def _relu(self, x): return np.maximum(0, x)
    def _relu_backward(self, gradient, pre_activation): return gradient * (pre_activation > 0)
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, X, is_training = True):
        self.X = X
        
        # Convulution pass
        self.conv_out = np.copy(self.conv_biases)
        for i in range(self.depth):
            for j in range(self.in_depth):
                self.conv_out[i] += signal.correlate2d(self.X[j], self.kernals[i, j], 'valid')
        
        self.activated_out = self._relu(self.conv_out)
        
        # FAST VECTORIZED MAX POOLING
        D, H, W = self.conv_output_shape
        P = self.pool_size
        
        # Reshape the feature map into 2x2 blocks using 6D array manipulation
        reshaped = self.activated_out.reshape(D, H // P, P, W // P, P)
        
        # Max along the block axes (axes 2 and 4)
        self.pooled_out = reshaped.max(axis=(2, 4))
        
        # Create a boolean mask of where the maximums were for backward pass
        replicated_max = self.pooled_out[:, :, np.newaxis, :, np.newaxis]
        self.pool_mask = (reshaped == replicated_max).reshape(D, H, W)
        self.flat_out = self.pooled_out.reshape(-1, 1)

        # Dropout
        if is_training:
            self.dropout_mask = np.random.rand(*self.flat_out.shape) >= self.dropout_rate
            self.flat_out = (self.flat_out * self.dropout_mask) / (1.0 - self.dropout_rate)

        # Dense layer
        self.dense_out = np.dot(self.dense_weights, self.flat_out) + self.dense_biases
        self.probabilities = self._softmax(self.dense_out)

        return self.probabilities

    def backward(self, target_label):
        loss_gradient = np.copy(self.probabilities)
        loss_gradient[target_label, 0] -= 1.0 

        # Dense gradients
        dense_weights_gradient = np.dot(loss_gradient, self.flat_out.T)
        dense_input_gradient = np.dot(self.dense_weights.T, loss_gradient)

        dense_input_gradient = (dense_input_gradient * self.dropout_mask) / (1.0 - self.dropout_rate)
        
        # Unflatten to pool shape
        pool_gradient = dense_input_gradient.reshape(self.pool_output_shape)
        
        # Unpool
        # e.g., [[G]] -> [[G, G], [G, G]]
        P = self.pool_size
        repeated_grad = np.repeat(np.repeat(pool_gradient, P, axis=1), P, axis=2)
        
        # Use our fast boolean mask to only keep gradients where the max was found
        conv_gradient_after_relu = repeated_grad * self.pool_mask
        
        # Conv gradients
        conv_gradient = self._relu_backward(conv_gradient_after_relu, self.conv_out)
        kernal_gradient = np.zeros(self.kernals_shape)
        input_gradient = np.zeros(self.in_shape)

        for i in range(self.depth):
            for j in range(self.in_depth):
                kernal_gradient[i, j] = signal.correlate2d(self.X[j], conv_gradient[i], 'valid')
                input_gradient[j] += signal.correlate2d(conv_gradient[i], self.kernals[i, j], 'full')
        
        # Updates
        self.dense_weights -= self.learning_rate * dense_weights_gradient
        self.dense_biases -= self.learning_rate * loss_gradient
        self.kernals -= self.learning_rate * kernal_gradient
        self.conv_biases -= self.learning_rate * conv_gradient

        return input_gradient