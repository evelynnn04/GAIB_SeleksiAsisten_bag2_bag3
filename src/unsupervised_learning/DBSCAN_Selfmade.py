import numpy as np
from collections import deque

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

class DBSCAN_Selfmade:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', p=3):
        '''
        eps: jarak maksimal biar bisa dianggep tetangga (secluster)
        min_samples: minimal anggota dalam satu cluster 
        metric: metric perhitungan jarak, in this case ada 3 (euclidean, manhattan, minskowski)
        p: parameter order buat minkowski
        '''
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.metric = metric
        self.p = p

    def fit(self, X):
        X = X.to_numpy() if not isinstance(X, np.ndarray) else X
        n_points = X.shape[0]
        self.labels = np.full(n_points, -1)
        cluster_id = 0

        for i in range(n_points):
            if self.labels[i] != -1:
                continue
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1

    def _region_query(self, X, point_idx):
        distances = self._calculate_distances(X, point_idx)
        return np.where(distances < self.eps)[0].tolist()

    def _calculate_distances(self, X, point_idx):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X - X[point_idx]), axis=1)
        elif self.metric == 'minkowski':
            return np.sum(np.abs(X - X[point_idx]) ** self.p, axis=1) ** (1 / self.p)

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        queue = deque(neighbors)

        while queue:
            neighbor_idx = queue.popleft()

            if self.labels[neighbor_idx] == -1:  
                self.labels[neighbor_idx] = cluster_id

            if self.labels[neighbor_idx] != -1:  
                continue

            self.labels[neighbor_idx] = cluster_id
            new_neighbors = self._region_query(X, neighbor_idx)

            if len(new_neighbors) >= self.min_samples:
                queue.extend(new_neighbors)

    def predict(self):
        return self.labels
