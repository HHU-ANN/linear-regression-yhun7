# 最终在main函数中传入一个维度为6的numpy数组，输出预测值
import os
try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def ridge(data, alpha=0.1):
    x, y = read_data()
    n_features = x.shape[1]
    a = np.eye(n_features)
    best_weight = np.zeros((n_features, 1))
    best_score = float('-inf')
    for i in range(100):
        weight = np.linalg.inv(x.T @ x + alpha * a) @ x.T @ y
        y_pred = x @ weight
        score = r2_score(y, y_pred)
        if score > best_score:
            best_score = score
            best_weight = weight
        alpha *= 0.9  # decrease regularization strength for next iteration
    return data @ best_weight

def lasso(data, alpha=0.1, lr=1e-12, max_iter=120000):
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