import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import TimeSeriesDataset
from module.model1 import TimeSeriesModel
from metric import calculate_metrics
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TimeSeriesModel(
    num_features=12, num_out_features=1, num_timesteps=300,
    prediction_ts_size=1, hidden_dim=128, num_layers=3
).to(device)

model.load_state_dict(torch.load('./trained_model/model_01.pth'))
model.eval()

data_csv_path = "./dataset/test.csv"  
history_size = 300 
interval = 10  
test_dataset = TimeSeriesDataset(data_csv_path=data_csv_path, history_size=history_size, forecast_size=1, interval=interval)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
criterion = nn.MSELoss()

total_test_loss = 0
total_metrics = {'MSE': 0, 'MAE': 0, 'RMSE': 0}
num_batches = 0

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = model(x_test)
        test_loss = criterion(outputs, y_test)
        total_test_loss += test_loss.item()

        # Calculate metrics
        batch_predictions = outputs.cpu().numpy()
        batch_labels = y_test.cpu().numpy()
        batch_metrics = calculate_metrics(batch_labels, batch_predictions)
        
        # Aggregate metrics
        for key in total_metrics:
            total_metrics[key] += batch_metrics[key]
        
        num_batches += 1

        # Print metrics and loss
        print(f"Test Loss: {test_loss.item():.4f}, MSE: {batch_metrics['MSE']:.4f}, MAE: {batch_metrics['MAE']:.4f}, RMSE: {batch_metrics['RMSE']:.4f}")

# Calculate average loss and metrics
avg_test_loss = total_test_loss / num_batches
for metric in total_metrics:
    total_metrics[metric] /= num_batches

print(f"\nAverage Test Loss: {avg_test_loss:.4f}")
for metric_name, value in total_metrics.items():
    print(f"Average {metric_name}: {value:.4f}")
