import numpy as np


'''
K Nearest Neighbors

The KNN class is a simple implementation of the K Nearest Neighbors algorithm. 
It is used to classify data points based on their similarity to other data points. 
The algorithm is as follows: 


Supported Algorith: Brute
TODO: Implement KD Tree, Ball Tree algorithms 




'''


class KNN:
    '''
    The KNN takes in the following parameters:
    k: The number of neighbors to use for classification.
    metric: The distance metric to use. The default is Euclidean distance.
    Other supported parameters are {manhattan, minkowski}.
    weights: The weight function to use. The default is uniform weights.
    Other supported parameters are {uniform, distance}
    p: The power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    algorithm: The algorithm to use for the nearest neighbors search. The default is brute force.
    Other supported parameters are {brute} 
    Other algorithms {ball_tree, kd_tree, auto} are not supported. They will be added in future versions.
    '''
    def __init__(self, k=3, p=2, metric='euclidean', weights='uniform', algorithm='brute'):
        self.k = k
        self.p = p
        self.metric = metric
        self.weights = weights
        self.algorithm = algorithm
        
    def _init_params(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)
        self.n_samples = len(self.X)
        self.n_features = len(self.X[0])

    def _euclidean_distance(self, x1, x2):
        '''
        Euclidean distance between two vectors x1 and x2.
        sqrt(sum((x1 - x2)^2))
        '''
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def _manhattan_distance(self, x1, x2):
        '''
        Manhattan distance between two vectors x1 and x2.
        sum(|x1 - x2|)
        '''
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1, x2):
        '''
        Minkowski distance between two vectors x1 and x2.
        (sum(|x1 - x2|^p))^(1/p)
        '''
        return np.sum(np.abs(x1 - x2)**self.p)**(1/self.p)
    
    def distance(self, x1, x2):
        '''
        Distance between two vectors x1 and x2.
        Appropriate distance metric is used based on the value of self.metric.
        '''
        if self.metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError('Invalid distance function')
    
    def _uniform_weights(self, distances):
        '''
        Uniform weights function.
        '''
        return np.ones_like(distances)
    
    def _distance_weights(self, distances):
        '''
        Distance weights function. 
        It is the inverse of the distance.
        '''
        return 1 / distances
    
    def weights(self, distances):
        '''
        Weights function.
        Selects the appropriate weights function based on the value of self.weights.
        '''
        if self.weights == 'uniform':
            return self._uniform_weights(distances)
        elif self.weights == 'distance':
            return self._distance_weights(distances)
        else:
            raise ValueError('Invalid weights function')
    
    def fit(self, X, y):
        '''
        Fit the KNN model using X as training data and y as target values.
        '''
        self._init_params(X, y)
        return self.__str__()
    
    def predict(self, X):
        '''
        Predict the class labels for the provided data.
        '''
        if self.algorithm == 'brute':
            return self._predict_brute(X).astype(np.int64)
        else:
            raise ValueError('Invalid algorithm')
        
    def _predict_brute(self, X):
        '''
        Predicts the class labels for the provided data using brute force.
        '''
        if self.weights == 'uniform':
            return self._predict_brute_uniform(X)
        elif self.weights == 'distance':
            return self._predict_brute_distance(X)
        else:
            raise ValueError('Invalid weights function')
    
    def _predict_brute_uniform(self, X):
        '''
        Predicts the class labels for the provided data using brute force and uniform weights.
        '''
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            distances = np.array([self.distance(x, x_train) for x_train in self.X])
            k_idx = np.argsort(distances)[:self.k]
            k_nearest_classes = self.y[k_idx]
            y_pred[i] = np.argmax(np.bincount(k_nearest_classes))
        return y_pred
    
    def _predict_brute_distance(self, X):
        '''
        Predicts the class labels for the provided data using brute force and distance weights.
        '''
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            distances = np.array([self.distance(x, x_train) for x_train in self.X])
            k_idx = np.argsort(distances)[:self.k]
            k_nearest_classes = self.y[k_idx]
            k_nearest_distances = distances[k_idx]
            y_pred[i] = np.argmax(np.bincount(k_nearest_classes, weights=k_nearest_distances))
        return y_pred
    
    def score(self, X, y):
        '''
        Returns the mean accuracy on the given test data and labels.
        '''
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)
    
    def __str__(self):
        return f'KNN(k={self.k}, p={self.p}, metric={self.metric}, weights={self.weights}, algorithm={self.algorithm})'
    
    def __len__(self):
        return self.n_samples

