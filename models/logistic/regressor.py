import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.01, lmbda=1, num_iter=1000, fit_intercept=True):
        """
        Inicializa o regressor com uma taxa de aprendizado alpha,
        número de iterações num_iter e adiciona feature para intercept
        caso fit_intercept == True.
        """
        self.alpha = alpha
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.lmbda = lmbda

    def __add_intercept(self, X):
        """
        Adiciona intercept ao array X.
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        """
        Retorna sigmoide de z.
        """
        return 1 / (1 + np.exp(-z))

    def __computeCost(self, h):
        """
        Retorna o custo J entre h(X), parametrizado por theta, e y.
        """
        a = -self.y @ np.log(h)
        b = (1 - self.y) @ np.log(1 - h)
        reg_term = (self.lmbda / (2 * self.y.size)) * (self.theta[1:] ^ 2).sum()
        return (a - b).mean() + reg_term

    def __gradient(self, h):
        """
        Retorna o gradiente da função de custo.
        """
        gradient = (self.X_.T @ (h - self.y_)) / self.y_.size
        gradient[1:] += (self.lmbda / self.y_.size) * self.theta[1:]
        return gradient

    def fit(self, X, y):
        """
        Computa os parâmetros theta para o melhor fit.
        """
        X, y = check_X_y(X, y)

        # Adiciona intercept caso fit_intercept == True.
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # Inicializa theta com zeros.
        self.theta = np.zeros(X.shape[1])

        # Adiciona arrays à classe (necessário p/ GridSearchCV)
        self.X_ = X
        self.y_ = y

        # Computa theta ótimo via método de gradiente.
        for i in range(self.num_iter):
            z = self.X_ @ self.theta
            h = self.__sigmoid(z)
            grad = self.__gradient(h)

            self.theta -= self.alpha * grad

        return self

    def predictProb(self, X):
        check_is_fitted(self)
        X = check_array(X)

        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(X @ self.theta)

    def predict(self, X, threshold=0.5):
        check_is_fitted(self)
        X = check_array(X)
        return self.predictProb(X) >= threshold
