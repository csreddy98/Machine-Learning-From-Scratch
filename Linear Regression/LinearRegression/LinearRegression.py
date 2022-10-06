# This file is part of Linear Regression.
# We are implementing Linear Regression using Gradient Descent Algorithm from scratch.

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        '''
        :param learning_rate: The step length that will be used when updating the weights
        :param epochs: Number of iterations or epochs
        '''
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost = None
        
    def _init_params(self, X, y):
        '''
        :param X: Feature matrix
        :param y: Actual values
        :return: Initialized weights and bias
        '''
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        self.cost = 0
        
    def h(self, X, theta):
        '''
        Hypothesis function
        :param X: Feature matrix
        :return: Predicted values
        '''
        return np.matmul(X, theta)

    def cost_function(self, X, y, theta):
        '''
        :param y: Actual values
        :param y_pred: Predicted values
        :return: Mean Squared Error
        '''
        return np.mean((y - self.h(X, theta)) ** 2)

    def gradient_descent(self, X, y, y_pred):
        '''
        :param X: Feature matrix
        :param y: Actual values
        :param y_pred: Predicted values
        :return: Updated weights and bias
        dw = -2 * X.T.dot(y - y_pred) / len(y)
        db = -2 * (y - y_pred).sum() / len(y)
        '''
        dw = -2 * X.T.dot(y - y_pred) / len(y)
        db = -2 * (y - y_pred).sum() / len(y)
        return dw, db
    
    def update_weights(self, dw, db):
        '''
        :param dw: Gradient of weights
        :param db: Gradient of bias
        :return: Updated weights and bias
        '''
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def fit(self, X, y):
        '''
        :param X: Feature matrix
        :param y: Actual values
        :return: Trained weights and bias
        '''
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            y_pred = self.predict(X)
            self.cost = self.cost_function(y, y_pred, self.weights)
            dw, db = self.gradient_descent(X, y, y_pred)
            self.update_weights(dw, db)
        
    def predict(self, X):
        '''
        :param X: Feature matrix
        :return: Predicted values
        '''
        return np.dot(X, self.weights) + self.bias
    
    def score(self, y, y_pred):
        '''
        :param y: Actual values
        :param y_pred: Predicted values
        :return: R2 score
        '''
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v
    
    def mean_absolute_error(self, y, y_pred):
        '''
        :param y: Actual values
        :param y_pred: Predicted values
        :return: Mean Absolute Error
        '''
        return np.mean(np.abs(y - y_pred))
    
    def mean_squared_error(self, y, y_pred):
        '''
        :param y: Actual values
        :param y_pred: Predicted values
        :return: Mean Squared Error
        '''
        return np.mean((y - y_pred) ** 2)
    
    def root_mean_squared_error(self, y, y_pred):
        '''
        :param y: Actual values
        :param y_pred: Predicted values
        :return: Root Mean Squared Error
        '''
        return np.sqrt(np.mean((y - y_pred) ** 2))
    
    def mean_absolute_percentage_error(self, y, y_pred):
        '''
        :param y: Actual values
        :param y_pred: Predicted values
        :return: Mean Absolute Percentage Error
        '''
        return np.mean(np.abs((y - y_pred) / y)) * 100     
