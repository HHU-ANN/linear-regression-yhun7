# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data, alpha=-0.1):
    x, y = read_data()
    a = np.eye(6)
    weight = np.dot(np.linalg.inv(np.dot(x.T, x) + alpha * a),  np.dot(x.T, y))
    return data @ weight

def lasso(data, alpha=2800, lr=1e-10, max_iter=100000):
    x, y = read_data()
    weight = data
    y_pred = np.dot(weight, x.T)
    for i in range(max_iter):
        y_pred = np.dot(weight, x.T)
        error = y_pred - y
        gradient = np.dot(error, x) + alpha * np.sign(weight)
        weight = weight * (1 - (lr * alpha / 6)) - gradient * lr
    return data @ weight


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y