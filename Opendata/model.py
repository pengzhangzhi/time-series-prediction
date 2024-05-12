import torch
import torch.nn as nn

class TimeSeriesModel(nn.Module):
    def __init__(self, num_features,num_out_features, num_timesteps, prediction_ts_size, hidden_dim,num_layers=5,nhead=4):
        super(TimeSeriesModel, self).__init__()
        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.num_out_features = num_out_features
        self.prediction_ts_size = prediction_ts_size

        # Time CNN
        conv_block = [
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        ] * 3
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_timesteps, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.layers = nn.ModuleList(
            [
                nn.Conv1d(in_channels=num_features, out_channels=hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                *conv_block,
                
            ]
        )

        self.out_layers = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_out_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Linear(self.num_timesteps,self.prediction_ts_size)
        )
        

    def forward(self, x):

        x = x.permute(0, 2, 1)
        
        for l in self.layers:
            x = l(x)
        # x shape: (batch, nd, t)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        
        x = x.permute(1,0,2)
        x = self.out_layers(x)

        return x

