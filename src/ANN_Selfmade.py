# source: https://github.com/lionelmessi6410/Neural-Networks-from-Scratch/blob/main/NN-from-Scratch.ipynb

# https://medium.com/@waleedmousa975/building-a-neural-network-from-scratch-using-numpy-and-math-libraries-a-step-by-step-tutorial-in-608090c20466

import numpy as np
import time

class ANN_Selfmade():
    def __init__(self, sizes, activation='sigmoid'):
        self.sizes = sizes
        
        # Choose activation function
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
        else:
            raise ValueError("Activation function is currently not support, please use 'relu' or 'sigmoid' instead.")
        
        # Save all weights
        self.params = self.initialize()
        # Save all intermediate values, i.e. activations
        self.cache = {}
    
    def custom_rounding(arr, threshold=0.5):
        """
        Rounds elements of a NumPy array based on a custom threshold.

        Parameters:
        arr (numpy.ndarray): Input array to round.
        threshold (float): The threshold for rounding. 
                        Values >= threshold round up, values < threshold round down.

        Returns:
        numpy.ndarray: Array with rounded values.
        """
        # Apply the rounding based on the threshold
        rounded_array = np.where(arr - np.floor(arr) > threshold, np.ceil(arr), np.floor(arr))
        
        return rounded_array

        
    def relu(self, x, derivative=False):
        '''
            Derivative of ReLU is a bit more complicated since it is not differentiable at x = 0
        
            Forward path:
            relu(x) = max(0, x)
            In other word,
            relu(x) = 0, if x < 0
                    = x, if x >= 0

            Backward path:
            ∇relu(x) = 0, if x < 0
                     = 1, if x >=0
        '''
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    def sigmoid(self, x, derivative=False):
        '''
            Forward path:
            σ(x) = 1 / 1+exp(-z)
            
            Backward path:
            ∇σ(x) = exp(-z) / (1+exp(-z))^2
        '''
        x = np.clip(x, -1*2**8, 2**8)
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        '''
            softmax(x) = exp(x) / ∑exp(x)
        '''
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialize(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_layer=self.sizes[1]
        output_layer=self.sizes[2]
        
        params = {
            "W1": np.random.randn(hidden_layer, input_layer) * np.sqrt(1./input_layer),
            "b1": np.zeros((hidden_layer, 1)) * np.sqrt(1./input_layer),
            "W2": np.random.randn(output_layer, hidden_layer) * np.sqrt(1./hidden_layer),
            "b2": np.zeros((output_layer, 1)) * np.sqrt(1./hidden_layer)
        }
        # print("params: ", params)
        return params
    
    def initialize_momemtum_optimizer(self):
        momemtum_opt = {
            "W1": np.zeros(self.params["W1"].shape),
            "b1": np.zeros(self.params["b1"].shape),
            "W2": np.zeros(self.params["W2"].shape),
            "b2": np.zeros(self.params["b2"].shape),
        }
        return momemtum_opt

    def feed_forward(self, x):
        '''
            y = σ(wX + b)
        '''
        # print(f"X = {x}")
        self.cache["X"] = x
        # print(f"Z1 = {self.params["W1"]} * {self.cache["X"]} + {self.params["b1"]}")
        self.cache["Z1"] = np.dot(self.params["W1"], self.cache["X"].T) + self.params["b1"]
        # print(f"Z1 = {self.cache["Z1"]}")
        self.cache["A1"] = self.activation(self.cache["Z1"])
        # print(f"Z2 = {self.params["W2"]} * {self.cache["A1"]} + {self.params["b2"]}")
        self.cache["Z2"] = np.dot(self.params["W2"], self.cache["A1"]) + self.params["b2"]
        # print(f"Z2 = {self.cache["Z2"]}")
        self.cache["A2"] = self.sigmoid(self.cache["Z2"])
        return self.cache["A2"]
    
    def back_propagate(self, y, output):
        '''
            Backpropagation for calculating the updates of the neural network's parameters.
        '''
        m = y.shape[0]
        
        # retrieve the intermediate values
        Z1 = self.cache["Z1"]
        A1 = self.cache["A1"]
        Z2 = self.cache["Z2"]
        A2 = self.cache["A2"]
        
        # compute the derivative of the loss with respect to A2
        epsilon = 1e-8
        dA2 = - (y/(A2 + epsilon)) + ((1-y)/(1-A2 + epsilon))

        
        # compute the derivative of the activation function of the output layer
        dZ2 = np.clip(dA2 * (A2 * (1-A2)), -1, 1)
        
        # compute the derivative of the weights and biases of the output layer
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # compute the derivative of the activation function of the hidden layer
        dA1 = np.dot(self.params["W2"].T, dZ2)
        dZ1 = dA1 * (A1 * (1-A1))
        
        # compute the derivative of the weights and biases of the hidden layer
        dW1 = (1/m) * np.dot(dZ1, self.cache['X'])
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        return self.grads 
        # current_batch_size = y.shape[0]

        # # Compute the derivative of the loss with respect to the output (dZ2)
        # dZ2 = output - y.T

        # # Compute gradients for the weights and biases of the output layer
        # dW2 = (1./current_batch_size) * np.matmul(dZ2, self.cache["A1"].T)
        # db2 = (1./current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)

        # # Compute the derivative of the activation function for the hidden layer (dZ1)
        # dA1 = np.matmul(self.params["W2"].T, dZ2)
        # dZ1 = dA1 * self.activation(self.cache["Z1"], derivative=True)

        # # Compute gradients for the weights and biases of the hidden layer
        # dW1 = (1./current_batch_size) * np.matmul(dZ1, self.cache["X"])
        # db1 = (1./current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

        # # Store the gradients in the grads attribute
        # self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        # print(f"gradients: {self.grads}")
        # return self.grads

    
    def cross_entropy_loss(self, y, output):
        '''
            L(y, ŷ) = −(1/m) * ∑y*log(ŷ)
            Cross-entropy loss calculation
        '''
        # Small constant to avoid log(0)
        epsilon = 1e-15

        # Clip output to avoid log(0) issues
        output = np.clip(output, epsilon, 1 - epsilon)

        # Compute the cross-entropy loss
        # l_sum = np.sum(y * np.log(output))
        m = y.shape[0]

        # Average the loss over all samples
        # l = -(1. / m) * l_sum
        l = -(1/m) * np.sum(y*np.log(output) + (1-y)*np.log(1-output))
        return l

                
    def optimize(self, l_rate=0.1, beta=.9):
        '''
            Stochatic Gradient Descent (SGD):
            θ^(t+1) <- θ^t - η∇L(y, ŷ)
            
            Momentum:
            v^(t+1) <- βv^t + (1-β)∇L(y, ŷ)^t
            θ^(t+1) <- θ^t - ηv^(t+1)
        '''
        if self.optimizer == "sgd":
            for key in self.params:
                # print(f"updating {key}: {key} = {self.params[key]} - {l_rate} * {self.grads[key]}")
                self.params[key] = self.params[key] - l_rate * self.grads[key]
        elif self.optimizer == "momentum":
            for key in self.params:
                # print(f"updateing {key}: {key} = {beta} * {self.momemtum_opt[key]} + (1. - {beta}) * {self.grads[key]}")
                self.momemtum_opt[key] = (beta * self.momemtum_opt[key] + (1. - beta) * self.grads[key])
                self.params[key] = self.params[key] - l_rate * self.momemtum_opt[key]
        else:
            raise ValueError("Optimizer is currently not support, please use 'sgd' or 'momentum' instead.")

    def accuracy(self, y, output):
        # return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))
        predictions = (output.T > 0.5).astype(int)
        return np.mean(predictions == y)


    def train(self, x_train, y_train, x_test, y_test, epochs=10, 
              batch_size=64, optimizer='momentum', l_rate=0.1, beta=.9):
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = -(-x_train.shape[0] // self.batch_size)
        '''
        Ini kaya pembagiannya gitu
        kenapa di - - ?
        Let say 103 // 32 = 3, padahal 32 * 3 = 96, jadi ada 7 row yg kelewat 
        Kalo - (-103 // 32) = - (-4) = 4 (actually it's around -3.2) 
        '''
        
        # Initialize optimizer
        self.optimizer = optimizer
        if self.optimizer == 'momentum':
            self.momemtum_opt = self.initialize_momemtum_optimizer()
        
        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        
        # Train
        for i in range(self.epochs):
            # Shuffle
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                # Batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                
                # Forward
                output = self.feed_forward(x)
                # output = self.feed_forward(x_train)
                # print("Output after feed_forward:", output)
                # Backprop
                _ = self.back_propagate(y, output)
                # Optimize
                self.optimize(l_rate=l_rate, beta=beta)

            # Evaluate performance
            # Training data
            output = self.feed_forward(x_train)
            # print(f"Before rounding: {output}")
            output = np.where(output - np.floor(output) > 0.5, np.ceil(output), np.floor(output))
            # print(f"After rounding: {output}")
            train_acc = self.accuracy(y_train, output)
            train_loss = self.cross_entropy_loss(y_train, output)
            # Test data
            output = self.feed_forward(x_test)
            test_acc = self.accuracy(y_test, output)
            test_loss = self.cross_entropy_loss(y_test, output)
            print(template.format(i+1, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))

    def predict(self, X):
        """
        Predict the output for the given input X.
        
        Parameters:
        X (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Predicted output.
        """
        # Perform feedforward on the input data
        output = self.feed_forward(X)
        
        # Apply custom rounding or thresholding for binary classification
        predictions = np.where(output.T > 0.5, 1, 0)
        
        return predictions