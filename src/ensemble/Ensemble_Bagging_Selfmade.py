'''
References:
- https://towardsdatascience.com/ensemble-learning-from-scratch-20672123e6ca
- youtube 
'''

import numpy as np

class Ensemble_Bagging_Selfmade:
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0):
        '''
        base_estimator: model yang dipake
        n_estimators: banyak model yang dipake
        max_samples: persentase sampel yang dipake (rangenya 0 - 1), kalo 1 berarti 100% aka dipake semua 
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []

    def fit(self, X, y):
        X = X.to_numpy()
        y = y.to_numpy()
        n_samples = X.shape[0]
        self.models = []
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(range(n_samples), size=int(self.max_samples * n_samples), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train model
            model = self.base_estimator()
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)

    def predict(self, X):
        X = X.to_numpy()
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        # Buat tiap data, dipredict pake tiap model 
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # Voting
        final_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)
        ''' 
        bincount: count occurences tiap integer 
        astype(int): convert ke int
        argmax: return indeks maksimum
        axis=1: diapply buat seluruh row 
        arr=predictions: targetnya dari predictions 

        jadi misal predictions = [0, 1, 0, 1, 1]
        bincount -> [2, 3] (2 buat 0 sama 3 buat 1)
        argmax -> 1 (berarti yang 3)
        Jadi resultnya 1

        Kenapa pake func ribet gini? Karena kalo dipisahin satu2 lama banget komputasinya wkwk
        '''
        return final_predictions


