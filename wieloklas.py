import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions

class Perceptron(object):

    def __init__(self, eta=0.025, n_iter=500):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1+ X.shape[1])
        # print(self.w_)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        # print(self.w_)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class Multiclass(object):

    def __init__(self, ppn1, ppn3):
        self.ppn1 = ppn1
        self.ppn3 = ppn3

    def predict(self, X):
        result = []
        for data in X:
            if self.ppn1.predict(data) == 1:
                result.append(0)
            elif self.ppn3.predict(data) == 1:
                result.append(1)
            else:
                result.append(2)

        return np.array(result)


def main():


    iris = datasets.load_iris()
    # print(iris)
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    y1 = y_train.copy()
    y2 = y_train.copy()
    y3 = y_train.copy()

    y1[(y1 != 0)] = -1
    y1[y1 == 0] = 1
    ppn1 = Perceptron()
    ppn1.fit(X_train, y1)

    y2[(y2 != 1)] = -1
    y2[y2 == 1] = 1
    ppn2 = Perceptron()
    ppn2.fit(X_train, y2)

    y3[(y3 != 2)] = -1
    y3[y3 == 2] = 1
    ppn3 = Perceptron()
    ppn3.fit(X_train, y3)

    multi = Multiclass(ppn1,ppn3)
    print(multi.predict(X_test))

    # labels = []
    # found = 0
    # for data in X_test:
    #     if ppn1.predict(data) == 1:
    #         labels.append(0)
    #         found+= 1
    #     if ppn2.predict(data) == 1:
    #         labels.append(1)
    #         found += 1
    #     if ppn3.predict(data) == 1:
    #         labels.append(2)
    #         found += 1
    #     else:
    #         labels.append('X')


    # print(*labels, sep = ", ")
    # print("Found {}/{} labels".format(found, len(X_test)))

    plot_decision_regions(X=X_test, y=y_test, classifier=multi)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
