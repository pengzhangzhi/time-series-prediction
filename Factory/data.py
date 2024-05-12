import pandas as pd
import torch
from torch.utils.data import Dataset
import random


class TimeSeriesDataset(Dataset):
    def __init__(self, data_csv_path="./dataset/factory.csv", history_size=600, forecast_size=1, interval=10):
        self.data = pd.read_csv(data_csv_path) 
        columns_to_drop = ["Time", "timestamp"]
        self.data = self.data.drop(columns=[col for col in columns_to_drop if col in self.data.columns], axis=1)
        self.data.dropna(axis=1, how='all', inplace=True)
        self.history_size = history_size
        self.forecast_size = forecast_size
        self.interval = interval
        self.data.dropna(inplace=True)
    
        self.data_mean = self.data.mean().to_numpy().reshape(1,1,-1)
        self.data_std = self.data.std().to_numpy().reshape(1,1,-1)

    def __len__(self):
        return len(self.data) - self.history_size - self.forecast_size + 1


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            indices = range(start, stop, step)
            return [self.get_single_item(i) for i in indices]
        else:
            return self.get_single_item(idx)

    def get_single_item(self, idx):
        if idx + self.history_size > self.__len__():
            idx = random.randint(0, self.__len__())
            return self.get_single_item(idx)
         # slicing
        start_idx = idx
        end_idx = idx + self.history_size
        x_data = self.data.iloc[start_idx:end_idx].to_numpy()
        
        start_idx = idx + self.history_size + self.interval
        end_idx = idx + self.history_size + self.interval + self.forecast_size
        y_data = self.data.iloc[start_idx:end_idx].values

        x_data = (x_data - self.data_mean.reshape(1, -1)) / self.data_std.reshape(1, -1)
        return torch.Tensor(x_data), torch.Tensor(y_data)[..., 0:1]#choose predict label column

    def reverse_scaling(self, y_pred):
        B,N,M = y_pred.shape
        std = self.data_std
        mean = self.data_mean
        y_pred = y_pred * std + mean
        return y_pred
