import torch
from torch.utils.data import DataLoader
from data import TimeSeriesDataset
from model import TimeSeriesModel
from metric import calculate_metrics
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import DataLoader, random_split


# Hyperparameters
num_features = 12
history_size = 336
num_out_features = 1
prediction_ts_size = 1
hidden_dim = 128
interval = 10
batch_size = 512
num_layers = 3
model_path = './trained_model/timeseries_model.pth'

# Setup CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = TimeSeriesModel(
    num_features=num_features, num_timesteps=history_size, num_out_features=num_out_features,
    prediction_ts_size=prediction_ts_size, hidden_dim=hidden_dim,
    num_layers=num_layers
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))

# Prepare test data
data_csv_path = "./dataset/factory_for_test.csv"
dataset = TimeSeriesDataset(data_csv_path=data_csv_path, history_size=history_size, forecast_size=prediction_ts_size, interval=interval)
train_size = int(0.7 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Testing
model.eval()
total_test_loss = 0
criterion = torch.nn.MSELoss() 

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


# with torch.no_grad():
#     for x_test, y_test in test_loader:
#         x_test, y_test = x_test.to(device), y_test.to(device)
#         outputs = model(x_test)
#         test_loss = criterion(outputs, y_test)
#         total_test_loss += test_loss.item()
#         print(f"Batch Test Loss: {test_loss.item():.4f}")

# avg_test_loss = total_test_loss / len(test_loader)
# print(f"Average Test Loss: {avg_test_loss:.4f}")
