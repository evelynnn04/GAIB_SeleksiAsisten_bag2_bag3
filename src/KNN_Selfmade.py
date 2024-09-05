import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=1)

def minkowski_distance(x1, x2, p=3):
    return np.sum(np.abs(x1 - x2) ** p, axis=1) ** (1 / p)

class KNN_Selfmade:
    def __init__(self, neighbors=3, metric='euclidean', p=3):
        '''
        neighbor: jumlah tetangga (N) yang nantinya bakal diambil top N tetangga paling deket buat dicari modusnya dalam case binary classification 
        metric: cara perhitungan jarak 
        p: parameter order buat minkowski 
        '''
        self.neighbors = neighbors
        self.metric = metric
        self.p = p
    
    def fit(self, X, y):
        self.X_train = X.to_numpy()
        self.y_train = y.to_numpy()
    
    def predict(self, X):
        X = X.to_numpy()
        distances = self._compute_distances(X)
        return np.array([self._predict_single(d) for d in distances])
    
    def _compute_distances(self, X):
        if self.metric == 'euclidean':
            return np.array([euclidean_distance(x, self.X_train) for x in X])
        elif self.metric == 'manhattan':
            return np.array([manhattan_distance(x, self.X_train) for x in X])
        elif self.metric == 'minkowski':
            return np.array([minkowski_distance(x, self.X_train, self.p) for x in X])
        else:
            raise ValueError(f"Unsupported metric! Please choose between 'euclidean', 'manhattan', or 'minkowski'!")
    
    def _predict_single(self, distances):
        top_n_indices = np.argsort(distances)[:self.neighbors]
        top_n_labels = self.y_train[top_n_indices]
        return Counter(top_n_labels).most_common(1)[0][0] # cari label paling banyak muncul 
