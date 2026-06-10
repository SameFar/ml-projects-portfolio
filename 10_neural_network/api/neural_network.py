import numpy as np

ol = {
    'Recommended Clothing Colors': [3,'mc'],
    'Avoid Clothing Colors': [3,'mc'],
    'Recommended Materials': [7,'mc'],
    'Recommended Patterns': [7,'mc'],
    'Recommended Jewelry Metal': [3,'mc'], 
    'Recommended Clothing Color Wheel Region': [3,'mc']
}

class NeuralNetwork:
    def __init__(self, learning_rate = 0.05, input_size = 20, outputs = ol):
        self.learning_rate = learning_rate
        
        # Get number of neurons in hidden layer based on the standard formula
        hidden_layer_dim = ((input_size * 2) // 3) + 26

        # INITIALIZATION for the shared ReLU hidden layer
        # Formula: std = sqrt(2 / input_size)
        self.weights = np.random.randn(input_size, hidden_layer_dim) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, hidden_layer_dim))

        self.Z1 = None
        self.A1 = None

        self.heads = {}
        for name, meta in outputs.items():
            num_outputs = meta[0]
            task_type = meta[1]
            
            # XAVIER / GLOROT INITIALIZATION for the output heads
            # Balances the playing field between 3-output and 7-output heads
            glorot_std = np.sqrt(2.0 / (hidden_layer_dim + num_outputs))
            
            self.heads[name] = {
                'weights' : np.random.randn(hidden_layer_dim, num_outputs) * glorot_std,
                'bias' : np.zeros((1, num_outputs)),
                'type' : task_type
            }
        

    def forward(self, X):
        self.Z1 = np.dot(X, self.weights) + self.bias
        self.A1 = self.relu(self.Z1)
        preds = {}

        for name, meta in self.heads.items():
            Z_head = np.dot(self.A1, meta['weights']) + meta['bias']
            
            if meta['type'] == 'mc':
                preds[name] = self.softmax(Z_head) 
            else:
                preds[name] = self.sigmoid(Z_head) 

        
        if self.learning_rate > 0.001:
            self.learning_rate -= 0.00001

        return preds

    def backward(self, X, y, preds):
        rows = X.shape[0]

        grades = np.zeros_like(self.A1)

        # Update weights of each head individually and make shared error
        for name, meta in self.heads.items():
            error = preds[name] - y[name]
            
            # Calculate how much to change weights and bias of each output layer
            dW_head = (1 / rows) * np.dot(self.A1.T, error)
            db_head = (1 / rows) * np.sum(error, axis=0, keepdims=True)
            
            # The Blame Bucket for hidden layers
            grades += np.dot(error, meta['weights'].T)
            
            # Gradient descent 
            meta['weights'] -= self.learning_rate * dW_head
            meta['bias'] -= self.learning_rate * db_head
            
        # Use the Blame Bucket™ to update your input layer weights
        dZ1 = grades * self.relu_derivative(self.Z1)
        dW1 = (1 / rows) * np.dot(X.T, dZ1)
        db1 = (1 / rows) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update shared parameters
        self.weights -= self.learning_rate * dW1
        self.bias -= self.learning_rate * db1



    def relu(self, x):
        '''Negative values turn into zero'''
        return np.maximum(0,x)
    
    def relu_derivative(self, x):
        '''Returns 0 or 1'''
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        ''''Returns probability of the guess'''
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    


