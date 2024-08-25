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
