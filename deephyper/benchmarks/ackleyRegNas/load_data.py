import os
import numpy as np
from deephyper.benchmarks.benchmark_functions_wrappers import ackley_
from deephyper.contrib import preprocessing

HERE = os.path.dirname(os.path.abspath(__file__))

np.random.seed(2018)

def load_data(dest=None):
    """
    Generate data for polynome_2 function.
    Returns Tuple of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """
    size = 10000
    prop = 0.85
    f, (a, b), _ = ackley_()
    d = b - a
    x = np.array([a + np.random.random(10) * d for i in range(size)])
    y = np.array([[f(v)] for v in x])

    # preprocessing
    print(f'preprocessing data set')
    data = preprocessing.classic(x, y)
    x = data[:, :-1]
    y = data[:, -1:]

    sep_index = int(prop * size)
    train_X = x[:sep_index]
    train_y = y[:sep_index]

    valid_X = x[sep_index:]
    valid_y = y[sep_index:]

    print(f'train_X shape: {np.shape(train_X)}')
    print(f'train_y shape: {np.shape(train_y)}')
    print(f'valid_X shape: {np.shape(valid_X)}')
    print(f'valid_y shape: {np.shape(valid_y)}')
    return (train_X, train_y), (valid_X, valid_y)

if __name__ == '__main__':
    load_data()
