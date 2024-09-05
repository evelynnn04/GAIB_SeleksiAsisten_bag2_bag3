import numpy as np

class KMeans_Selfmade:
    def __init__(self, n_clusters=3, max_iter=100):
        '''
        n_cluster: banyaknya cluster yang ingin dibentuk
        max_iter: banyaknya iterasi (auto berhenti kalau centroid baru dan centroid lamanya sama)
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def initialize_centroids(self, X):
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices] # ambil titik random 

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, X):
        self.initialize_centroids(X.to_numpy())
        for i in range(self.max_iter):
            labels = self.assign_clusters(X.to_numpy())
            new_centroids = self.update_centroids(X, labels)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        return self.assign_clusters(X.to_numpy())

'''
1. Ambil titik random ajah sesuai banyak clusternya (in binary case berarti ambil 1 titik) -> centroid 
2. Classify based on centroid tadi 
3. Tentuin centroid baru dari mean per cluster 
4. Ulangin 2 sama 3 sampe clusternya ga berubah ato sampe max iter
'''