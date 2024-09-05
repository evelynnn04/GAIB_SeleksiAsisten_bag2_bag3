import numpy as np
import pandas as pd

def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities**2)

def information_gain(y, y_left, y_right):
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)
    
    return gini(y) - (p_left * gini(y_left) + p_right * gini(y_right))

def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold # semua x yg kurang dari threshold
    right_mask = ~left_mask # lainnya 
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask] # dataframe yg isinya true false

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class CART_Selfmade:
    def __init__(self, max_depth=None, min_samples_split=2):
        '''
        max_depth: maksimal kedaleman cabangnya (atau sampai gini impuritynya 0)
        min_sample_split: minimal ada berapa data biar bisa displit
        '''
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X.values, y.values)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Ini kalo udah ga nyabang lagi
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y, n_features)
        
        if best_feature is None: # udah gabisa displit 
            leaf_value = self._most_common_label(y) # berarti kalo masuk leaf ini labelnya modusnya 
            return Node(value=leaf_value)

        X_left, y_left, X_right, y_right = split_dataset(X, y, best_feature, best_threshold)
        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)
        return Node(best_feature, best_threshold, left_child, right_child)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_index, split_threshold = None, None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index]) # cari yg unique buat jadi threshold
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0: # datanya masuk ke salah satu cabang semua 
                    continue

                gain = information_gain(y, y_left, y_right) # seberapa banyak impuritynya berkurang 
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold # dapet split2annya 

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X.values])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

'''
Intinya di tiap cabang, dia cari split2an paling bagus (gainnya paling gede), terus direturn, terus nanti grow tree ke kiri sama kanan based on split2annya tadi.
Buat ngepredict tinggal ditraverse yeay kelar
'''