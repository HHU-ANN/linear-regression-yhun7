# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data, alpha=0):
    x, y = read_data()
    n_features = x.shape[1]
    a = np.eye(n_features)
    weight = np.dot(np.linalg.inv(np.dot(x.T, x) + alpha * a),  np.dot(x.T, y))
    return data @ weight

def lasso(data, alpha=0.01, lr=1e-15, max_iter=150000):
    x, y = read_data()
    n_sample, n_features = x.shape
    weight = np.zeros((n_features, 1))
    for i in range(max_iter):
        y_pred = x @ weight
        error = y - y_pred
        gradient = -2 * (x.T @ error) + alpha * np.sign(weight)
        weight -= lr * gradient.mean(axis=1, keepdims=True)
    return data @ weight


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y