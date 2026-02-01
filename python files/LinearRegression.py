import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weight) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - Y))
            db = (1 / n_samples) * np.sum(y_pred - Y)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):

        return np.dot(X, self.weight) + self.bias
    

    if __name__ ="__main__":
        #import
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn import datasets

    def mean_square_ error (y_true,y)_pred):

        return np.mean((y_true,y_pred)^^2)

