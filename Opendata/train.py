import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data import TimeSeriesDataset
from model import TimeSeriesModel
import matplotlib.pyplot as plt
import wandb
from metric import calculate_metrics
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


num_features=7
history_size=336
num_out_features = 1
prediction_ts_size=1
hidden_dim=128
interval=96
data_csv_path="./dataset/ETT-small/ETTh1.csv"

lr=0.0001
num_epochs = 100
batch_size = 512
num_layers = 3


# # start a new wandb run
# wandb.init(
#     project="opendata",
#     name="ETTh1_interval=96",
#     tags=["ETTh1","interval=96","opendata"],
#     )
# config= {
#     "learning_rate": lr,
#     "architecture": "transformer",
#     "dataset": data_csv_path,
#     "epochs": num_epochs,
#     "batch_size":batch_size,
#     "num_features":num_features,
#     "history_size":history_size,
#     "prediction_ts_size":prediction_ts_size,
#     "hidden_dim":hidden_dim,
#     "interval":interval
# }



model = TimeSeriesModel(
    num_features=num_features, num_timesteps=history_size, num_out_features=num_out_features,
    prediction_ts_size=prediction_ts_size, hidden_dim=hidden_dim,
    num_layers=num_layers
    ).to("cuda")

#checkpoint = torch.load('out1_v0.pth')
#model.load_state_dict(checkpoint)

dataset = TimeSeriesDataset(data_csv_path=data_csv_path, history_size=history_size, forecast_size=prediction_ts_size, interval=interval)

# Splitting the dataset
train_size = int(0.7 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
loss_lst = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    count = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1

    # Validation
    model.eval()
    total_valid_loss = 0
    val_count = 0
    with torch.no_grad():
        outputs_list, y_val_list = [], []
        for x_val, y_val in valid_loader:
            x_val, y_val = x_val.to("cuda"), y_val.to("cuda")
            outputs = model(x_val)
            # Accumulate for metric calculation
            outputs_list.append(outputs.cpu())
            y_val_list.append(y_val.cpu())
            # Compute loss
            val_loss = criterion(outputs.cpu(), y_val.cpu())
            total_valid_loss += val_loss.item()
            val_count += 1  
    avg_val_loss = total_valid_loss / val_count
    avg_loss = total_loss/count
    loss_lst.append(avg_loss)
    
    # Concatenate
    preds = np.concatenate(outputs_list, axis=0)
    trues = np.concatenate(y_val_list, axis=0)
    # Calculate metrics
    mae, mse, rmse, mape, mspe, rse, corr ,smape = calculate_metrics(preds, trues)
    print('mse:{}, mae:{},rmse{},smape{}'.format(mse, mae,rmse, smape))
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss/count} ")
    #wandb.log({"train_loss": avg_loss, "val_loss": avg_val_loss,"mse":mse,"mae":mae,"rmse":rmse,"smape":smape})
    torch.save(model.state_dict(), 'timeseries_model.pth')

# Testing
model.eval()
total_test_loss = 0
with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to("cuda"), y_test.to("cuda")
        outputs = model(x_test)
        test_loss = criterion(outputs.cpu(), y_test.cpu())
        total_test_loss += test_loss.item()
#wandb.log({"test_loss": total_test_loss/len(test_loader)})
print(f"Test Loss: {total_test_loss/len(test_loader)}")

# Save the model
torch.save(model.state_dict(), 'timeseries_model.pth')
