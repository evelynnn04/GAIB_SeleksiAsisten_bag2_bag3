import numpy as np
from collections import deque

class DBSCAN_Selfmade:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps # Ini jaraknya (euclidean)
        self.min_samples = min_samples # Minimal node satu cluster
        self.labels = None 

    def fit(self, X):
        n_points = X.shape[0]
        self.labels = np.full(n_points, -1)  # Initiate jadi -1 semua
        cluster_id = 0

        for i in range(n_points):
            if self.labels[i] != -1:  # Uda masuk cluster orang
                continue
            neighbors = self._region_query(X.to_numpy(), i) # Cari temen secluster
            if len(neighbors) < self.min_samples: # Noise
                self.labels[i] = -1  
            else:
                self._expand_cluster(X.to_numpy(), i, neighbors, cluster_id)
                cluster_id += 1

    def _region_query(self, X, point_idx):
        neighbors = []
        for i in range(X.shape[0]):
            if np.linalg.norm(X[point_idx] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        queue = deque(neighbors)

        while queue:
            neighbor_idx = queue.popleft()

            if self.labels[neighbor_idx] == -1: 
                self.labels[neighbor_idx] = cluster_id
            else:  
                continue

            self.labels[neighbor_idx] = cluster_id
            new_neighbors = self._region_query(X, neighbor_idx)

            if len(new_neighbors) >= self.min_samples:
                queue.extend(new_neighbors)

    def predict(self):
        return self.labels
