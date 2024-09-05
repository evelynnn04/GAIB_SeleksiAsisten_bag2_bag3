'''
References:
- https://medium.com/@aneesha161994/loss-functions-of-classification-models-90354b14db93
'''

import numpy as np

class Logistic_Regression_Selfmade:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_param=0.01, loss_function='cross_entropy', alpha=0.25, gamma=2):
        '''
        learning rate: laju belajarnya buat ngontrol seberapa besar perubahan weight dan bias 
        n_iteration: jumlah iterasi 
        regularization: nambahin penalti buat ngurangin overfitting
        reg_lamda: koefiisien penalti buat ngontrol seberapa besar penalti pada weight di proses regularization 
        loss function: buat ngitung kesalahan antara y_pred dan y_test 
        '''
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.loss_function = loss_function
        self.alpha = alpha # buat focal
        self.gamma = gamma # buat focal
        self.weights = None
        self.bias = None
    
    def sigmoid(self, x):
        x = np.clip(x, -(2**4-1), 2**4-1) # buat ngebatesin biar ga warning terus wkwk
        return 1 / (1 + np.exp(-x)) # rumusnya emang gini wkwk
    
    def compute_loss(self, y, y_predicted):
        epsilon = 1e-10 # Ini biar ga log(0) biar lossnya ga inf
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon) # buat ngebatesin biar ga warning terus, hasilnya juga lebih bagus wkwk
        if self.loss_function == 'cross_entropy':
            return -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        
        elif self.loss_function == 'focal': # ini cocok buat imbalanced data
            y_predicted = np.clip(y_predicted, epsilon, 1.0 - epsilon)
            pt = y * y_predicted + (1 - y) * (1 - y_predicted)
            focal_loss = -self.alpha * (1 - pt) ** self.gamma * np.log(pt)
            return np.mean(focal_loss)
        
        elif self.loss_function == 'logit':
            y_predicted = np.clip(y_predicted, epsilon, 1.0 - epsilon)
            logit_loss = - (y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
            return np.mean(logit_loss)
        else:
            raise ValueError("Unsupported loss function! Choose between 'cross_entropy', 'focal', 'logit'!")
    
    def add_regularization(self, loss): 
        # Regularization intinya buat ngurangin weight (y = x . weight + e) biar kalo ada kenaikan/penurunan x, y nya ga tbtb loncat wkwk
        if self.regularization == 'l2': # ridge
            loss += (self.lambda_param / len(self.y_train)) * self.weights
        elif self.regularization == 'l1': # lasso
            loss += (self.lambda_param / len(self.y_train)) * np.sign(self.weights)
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
   