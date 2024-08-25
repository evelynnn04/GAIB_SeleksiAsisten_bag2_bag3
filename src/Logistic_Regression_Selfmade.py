import numpy as np
import pandas as pd

class Logistic_Regression_Selfmade:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, reg_lambda=0.01, loss_function='cross_entropy'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.loss_function = loss_function
        self.weights = None
        self.bias = None
    
    def sigmoid(self, x):
        x = np.clip(x, -(2**4-1), 2**4-1) # buat ngebatesin biar ga warning terus wkwk, hasilnya juga lebih bagus aneh bgt wkwk (NANTI TANYAIN DEH)
        return 1 / (1 + np.exp(-x)) # rumusnya emang gini wkwk
    
    def compute_loss(self, y, y_predicted):
        epsilon = 1e-10 # Ini biar ga log(0) biar lossnya ga inf
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon) # buat ngebatesin biar ga warning terus wkwk
        if self.loss_function == 'cross_entropy':
            return -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        elif self.loss_function == 'mse':
            return np.mean((y - y_predicted) ** 2)
        # TODO: Tambahin MAE, RMSE biar dapet bonus wkwk
        else:
            raise ValueError("Unsupported loss function! Choose between 'cross_entropy' or 'mse'!")
    
    def add_regularization(self, loss): 
        # Regularization intinya buat ngurangin weight (y = x . weight + e) biar kalo ada kenaikan/penurunan x, y nya ga tbtb loncat wkwk
        if self.regularization == 'l2': # ridge
            loss += (self.reg_lambda / len(self.y_train)) * self.weights
        elif self.regularization == 'l1': # lasso
            loss += (self.reg_lambda / len(self.y_train)) * np.sign(self.weights)
        return loss
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.y_train = y
        
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            if self.regularization is not None:
                dw = self.add_regularization(dw)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if _ % 100 == 0:
                loss = self.compute_loss(y, y_predicted)
                print(f'Iteration {_}, Loss: {loss}')
    
    def predict_proba(self, X):
        y = np.dot(X, self.weights) + self.bias
        return self.sigmoid(y)
    
    def predict(self, X, threshold=0.5):
        y_predicted_proba = self.predict_proba(X)
        y_predicted = [1 if i > threshold else 0 for i in y_predicted_proba]
        return np.array(y_predicted)
   