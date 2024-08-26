import numpy as np

class SVC_Selfmade:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='linear'):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel_type = kernel
        self.w = None
        self.b = None
    
    def _linear_kernel(self, X):
        return X
    
    def _polynomial_kernel(self, X, degree=3):
        return np.power(X, degree)
    
    def _rbf_kernel(self, X, gamma=None):
        if gamma is None:
            gamma = 1 / X.shape[1]
        return np.exp(-gamma * np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2))
    
    def _compute_kernel(self, X):
        if self.kernel_type == 'linear':
            return self._linear_kernel(X)
        elif self.kernel_type == 'polynomial':
            return self._polynomial_kernel(X)
        elif self.kernel_type == 'rbf':
            return self._rbf_kernel(X)
        else:
            raise ValueError("Unsupported kernel type")
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features) # In my case self.w jadi [0, 0] soalnya binary 
        self.b = 0
        
        y_ = np.where(y <= 0, -1, 1) # Yg 0 diconvert jadi -1, ini masalah algoritma aja sii
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]
                
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)