import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def SMAPE(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def calculate_metrics(y_true, y_pred):
    metrics = {
        'MSE': mse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
    }
    return metrics
