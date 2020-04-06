import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


class Multiclass(object):
    def __init__(self, cls1, cls3):
        self.cls1 = cls1
        self.cls3 = cls3

    def predict(self, X):
        result = []
        for data in X:
            if self.cls1.predict(data) == 1:
                result.append(0)
            elif self.cls3.predict(data) == 1:
                result.append(1)
            else:
                result.append(2)


        return np.array(result)

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [1, 3]]
    y = iris.target
    y1 = y.copy()
    y2 = y.copy()
    y3 = y.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


    y1[(y1 != 0)] = -3
    y1[y1 == 0] = 1
    y1[y1 == -3] = 0

    y3[(y3 != 2)] = -3
    y3[y3 == 2] = 1
    y3[y3 == -3] = 0

    #w regresji logarytmicznej wyjście przyjmuje wartości 0 lub 1 (prawdopodobieństwa)
    lrgd1 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd1.fit(X, y1)

    lrgd3 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd3.fit(X, y3)


    multi = Multiclass(lrgd1, lrgd3)
    print(multi.predict(X_test))

    print(lrgd1.predict(X_test))

    plot_decision_regions(X=X_test, y=y_test, classifier=multi)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
