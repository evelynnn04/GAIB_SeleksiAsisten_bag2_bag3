import numpy as np
import pandas as pd
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p=3):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

class KNN_Selfmade:
    def __init__(self, neighbors=3, metric='euclidean', p=3):
        self.neighbors = neighbors
        self.metric = metric
        self.p = p  # ini buat minkowski
    
    def fit(self, X, y):
        self.X_train = X.to_numpy()
        self.y_train = y.to_numpy()
    
    def predict(self, X):
        y_pred = []
        for row_x in X.to_numpy():
            y_pred.append(self._predict(row_x))
        return np.array(y_pred)
    
    def _predict(self, x):
        arr = {}
        if self.metric == 'euclidean':
            for i in range(len(self.X_train)):
                index = i
                distance = euclidean_distance(x, self.X_train[i])
                arr[index] = distance
        elif self.metric == 'manhattan':
            for i in range(len(self.X_train)):
                index = i
                distance = manhattan_distance(x, self.X_train[i])
                arr[index] = distance
        elif self.metric == 'minkowski':
            for i in range(len(self.X_train)):
                index = i
                distance = minkowski_distance(x, self.X_train[i])
                arr[index] = distance
        else:
            raise ValueError(f"Unsupported distance! Please choose between 'euclidean', 'manhattan', or 'minkowski'!")
        
        top3_arr = sorted(arr.items(), key=lambda item: item[1])[0:self.neighbors]
        top_n_index = [x[0] for x in top3_arr]
        top_n_labels = [self.y_train[i] for i in top_n_index]
        return Counter(top_n_labels).most_common(1)[0][0]

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
    
class Gaussian_Naive_Bayes_Selfmade:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0, ddof=1) + 1e-9 # ditambah dikit biar ga 0 soalnya nanti kan jadi denominator ntar jadi NaN
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator # ini pdf kalo di rumusnya
    
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X.to_numpy()])

