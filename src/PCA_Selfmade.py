import numpy as np

class PCA_Selfmade:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        # Step 1: Standardize the data (mean = 0, variance = 1)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(X, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Step 4: Sort the eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top n_components eigenvectors (principal components)
        self.components = eigenvectors[:, :self.n_components]

        # Step 6: Calculate the explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance

    def transform(self, X):
        # Step 1: Standardize the data
        X = X.to_numpy()
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Step 2: Project the data onto the principal components
        return np.dot(X, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
