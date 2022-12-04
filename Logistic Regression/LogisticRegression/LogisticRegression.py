import numpy as np
from numpy.random import random

"""
Implementation of Logistic Regression

penalty:
l1: L1 regularization
l2: L2 regularization
elasticnet: Elastic Net regularization
none: No regularization


solver:
newton-cg: Newton-CG algorithm (only for L2 regularization or no regularization)
lbfgs: L-BFGS algorithm (only for L2 regularization or no regularization)
liblinear: Liblinear algorithm (only for L1 or l2 regularization)
sag: Stochastic Average Gradient descent algorithm (only for L2 regularization or no regularization)
saga: SAGA algorithm (for L1 or L2 regularization or Elastic Net regularization or no regularization)

"""

class LogisticRegression:
    def __init__(self, penalty='l2', solver='lbfgs', C=1.0, max_iter=100, tol=1e-4, random_state=None):
        self.penalty = penalty
        self.solver = solver
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
    def _init_params(self, n_features):
        self.coef_ = np.random.normal(size=(n_features, 1))
        self.intercept_ = np.random.normal(size=1)
        if self.penalty == 'l1':
            self.lam = 1 / (2 * self.C)
        elif self.penalty == 'l2':
            self.lam = 1 / self.C
        elif self.penalty == 'elasticnet':
            self.lam = 1 / self.C
        else:
            self.lam = 0
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, X, y):
        self._init_params(X.shape[1])
        if self.solver == 'newton-cg' or self.solver == 'lbfgs':
            self._fit_newton(X, y)
        elif self.solver == 'liblinear':
            self._fit_liblinear(X, y)
        elif self.solver == 'sag':
            self._fit_sag(X, y)
        elif self.solver == 'saga':
            self._fit_saga(X, y)
        else:
            raise ValueError("The solver %s is not supported" % self.solver)
        return self

    def _fit_newton(self, X, y):
        for i in range(self.max_iter):
            z = X @ self.coef_ + self.intercept_
            p = 1 / (1 + np.exp(-z))
            grad = X.T @ (p - y) + self.lam * self.coef_
            hess = X.T @ np.diag(p.ravel() * (1 - p.ravel())) @ X + self.lam * np.eye(X.shape[1])
            delta = np.linalg.inv(hess) @ grad
            self.coef_ -= delta[:X.shape[1]]
            self.intercept_ -= delta[X.shape[1]]
            if np.linalg.norm(delta) < self.tol:
                break
        
    def _fit_liblinear(self, X, y):
        for i in range(self.max_iter):
            z = X @ self.coef_ + self.intercept_
            p = 1 / (1 + np.exp(-z))
            grad = X.T @ (p - y) + self.lam * np.sign(self.coef_)
            hess = X.T @ np.diag(p.ravel() * (1 - p.ravel())) @ X + self.lam * np.eye(X.shape[1])
            delta = np.linalg.inv(hess) @ grad
            self.coef_ -= delta[:X.shape[1]]
            self.intercept_ -= delta[X.shape[1]]
            if np.linalg.norm(delta) < self.tol:
                break
            
    def _fit_sag(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        grad = np.zeros((n_samples, n_features + 1))
        for i in range(self.max_iter):
            z = X @ self.coef_ + self.intercept_
            p = 1 / (1 + np.exp(-z))
            grad[:, :n_features] = X * (p - y).reshape(-1, 1)
            grad[:, n_features] = p - y
            grad += self.lam * np.r_[self.coef_.T, self.intercept_]
            delta = np.mean(grad, axis=0)
            self.coef_ -= delta[:X.shape[1]]
            self.intercept_ -= delta[X.shape[1]]
            if np.linalg.norm(delta) < self.tol:
                break
    
    def _fit_saga(self, X, y): 
        n_samples = X.shape[0]
        n_features = X.shape[1]
        grad = np.zeros((n_samples, n_features + 1))
        for i in range(self.max_iter):
            z = X @ self.coef_ + self.intercept_
            p = 1 / (1 + np.exp(-z))
            grad[:, :n_features] = X * (p - y).reshape(-1, 1)
            grad[:, n_features] = p - y
            grad += self.lam * np.r_[self.coef_.T, self.intercept_]
            delta = np.mean(grad, axis=0)
            self.coef_ -= delta[:X.shape[1]]
            self.intercept_ -= delta[X.shape[1]]
            if np.linalg.norm(delta) < self.tol:
                break
            
    def predict_proba(self, X):
        
        z = X @ self.coef_ + self.intercept_
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        return np.where(self.predict_proba(X) > 0.5, 1, 0)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)