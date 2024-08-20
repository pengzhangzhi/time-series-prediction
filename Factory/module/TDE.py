import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class Trend(Dataset):
    # Time series dataset with trends
    def __init__(self, data_csv_path, history_size=600, forecast_size=1, interval=10, ma_period=5):
        self.data = pd.read_csv(data_csv_path)
        self.data = self.data.drop(["Time", "timestamp"], axis=1) 
        self.data.dropna(axis=1, how='all', inplace=True)
        self.history_size = history_size
        self.forecast_size = forecast_size
        self.interval = interval
        self.ma_period = ma_period
        self.data.dropna(inplace=True)
        # Z-score Scaling
        self.data_mean = self.data.mean().to_numpy().reshape(1, -1)
        self.data_std = self.data.std().to_numpy().reshape(1, -1)

    def __len__(self):
        # Adjust length
        return len(self.data) - self.history_size - self.forecast_size - self.ma_period + 2

    def __getitem__(self, idx):
        # Get single item
        if isinstance(idx, slice):
            # Get multiple items
            start, stop, step = idx.indices(len(self))
            # Generate indices
            indices = range(start, stop, step)
            # Get items
            return [self.get_single_item(i) for i in indices]
        else:
            return self.get_single_item(idx)
    # Get single item
    def get_single_item(self, idx):
        if idx + self.history_size + self.ma_period - 1 > self.__len__():
            # Randomly select an index
            idx = random.randint(0, self.__len__())
            return self.get_single_item(idx)
        # Get data
        start_idx = idx
        end_idx = idx + self.history_size + self.ma_period - 1  # Adjust index
        x_data = self.data.iloc[start_idx:end_idx].to_numpy()
        
        # Compute moving average
        moving_averages = np.array([np.convolve(x_data[:, i], np.ones(self.ma_period)/self.ma_period, mode='valid') for i in range(x_data.shape[1])]).T
        # Compute trends
        trends = np.diff(moving_averages, axis=0)
        trends = np.vstack([np.zeros((1, trends.shape[1])), trends])  # Pad the first difference

        x_data = (x_data[self.ma_period-1:] - self.data_mean) / self.data_std  # Align lengths
        
        # Concatenate x_data with trends
        trends = trends.reshape(-1, x_data.shape[1], 1)
        x_data = x_data.reshape(-1, x_data.shape[1], 1)
        x_data = np.concatenate([x_data, trends], axis=-1)  # Concatenate
        # Get y_data
        start_idx = idx + self.history_size + self.interval
        end_idx = start_idx + self.forecast_size
        y_data = self.data.iloc[start_idx:end_idx].values
        # Return as tensors
        return torch.Tensor(x_data), torch.Tensor(y_data)[..., 0:1]

    def reverse_scaling(self, y_pred):
        # Reverse scaling
        B, N, M = y_pred.shape
        std = self.data_std
        mean = self.data_mean
        y_pred = y_pred * std + mean
        return y_pred
