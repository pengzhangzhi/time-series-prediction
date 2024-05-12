import torch
import torch.nn as nn


class TimeSeriesModel(nn.Module):
    def __init__(self, num_features, num_out_features, num_timesteps, prediction_ts_size, hidden_dim,num_layers=5,nhead=8):
        super(TimeSeriesModel, self).__init__()
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.num_out_features = num_out_features
        

        conv_block = [
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        ] * 3
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.layers = nn.ModuleList(
            [
                nn.Conv1d(in_channels=num_timesteps, out_channels=hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                *conv_block,
                
            ]
        )

        self.out_layers = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=prediction_ts_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(self.num_features,self.num_out_features)
        )
        

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = x.permute(0,2,1)
        x = self.transformer_encoder(x)
        x = x.permute(0,2,1)
        x = self.out_layers(x)
        return x

