import numpy as np

class SVC_Selfmade:
    def __init__(self, learning_rate=0.001, lambda_param=0.1, n_iterations=1000, kernel='linear', threshold=0.05):
        '''
        learning rate: ini buat ngontrol penambahan weightnya (nanti kan dikaliin sama weightnya, kalo lrnya gede berarti penambahan weightnya juga langsung wuzz gicu). lr gede bikin model cepet konvergen tapi bisa bikin missing optimal point. 
        lamda_param: ini buat ngontrol penalti 
        n_iterations: iterasinya berapa kali 
        kernel: ini buat mapping datanya ke dimensi yang lebih tinggi 
        threshold: ambang batas buat nentuin prediksinya 0 atau 1
        '''
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.kernel_type = kernel
        self.threshold = threshold 
        self.w = None
        self.b = None
    
    def _linear_kernel(self, X): # Ini jadi 2D
        return X
    
    def _polynomial_kernel(self, X, degree=3): # Ini juga jadi 2D
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
            raise ValueError("Unsupported kernel! Please choose between 'linear', 'polynomial' or 'rbf'")
    
    def fit(self, X, y):
        X = X.values
        y = y.values
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Label encoding
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.n_iterations):
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
        K = self._compute_kernel(X) 
        approx = np.dot(K, self.w) - self.b
        return [1 if x >= self.threshold else 0 for x in approx]
    

'''
Jadi ini inti kodenya tuh bikin persamaan hyperplane, terus dinaikin dimensinya pake kernel
'''