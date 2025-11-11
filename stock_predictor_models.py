import torch
import torch.nn as nn
import torch.optim as optim
class CustomMinMaxScaler():
    def __init__(self):
        self.min = None
        self.max = None
    def fit(self, tensor):
        min_train = torch.min(torch.min(tensor, dim=0)[0], dim=0)[0]
        max_train = torch.max(torch.max(tensor, dim=0)[0], dim=0)[0]
        self.min = min_train
        self.max = max_train
    def transform(self, tensor):
        if self.min is None or self.max is None:
            raise Exception("Scaler must be fitted prior to transform")
        return (tensor - self.min) / (self.max - self.min)
    def fit_transform(self, tensor):
        self.fit(tensor)
        return self.transform(tensor)
    
    def inverse_transform(self, tensor):
        if self.min is None or self.max is None:
            raise Exception("Scaler must be fitted prior to inverse transform")
        return tensor * (self.max - self.min) + self.min

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockTransformer(nn.Module):
    def __init__(
        self, 
        input_dim=1, 
        hidden_dim=64, 
        nhead=4, 
        num_layers=2, 
        dim_feedforward=128,
        output_dim=1,
        dropout=0.1 
    ):
        super(StockTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is shape (batch_size, seq_length, input_dim)
        x = self.input_proj(x)
        x = self.transformer(x) # batch_size, seq_length, hidden_dim
        x = x[:, -1, :]
        out = self.fc_out(x) # batch_size, output_dim
        return out
