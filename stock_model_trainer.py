import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from fetch_stock_data import collect_data
from stock_predictor_models import StockLSTM, CustomMinMaxScaler, StockTransformer
import torch.optim as optim

class ModelTrainer():
    def __init__(self, criterion, model, optimizer, X_train, y_train):
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.X_train = X_train
        self.y_train = y_train
        self.losses = None

    def train(self, num_epochs):
        losses = []
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = self.criterion(output, self.y_train)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")
        self.losses = losses

    def plot_losses(self, showfig=False):
        if self.losses is None:
            raise Exception("Must train a model prior to plotting losses")
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Model Training Curve")
        if showfig:
            plt.show()
        else:
            plt.savefig("outputs/model_losses.png")
            plt.clf()

    
    def eval(self, X_test, y_test, scaler, showfig=False):
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(X_test)
        predicted_prices = scaler.inverse_transform(predicted)
        actual_prices = scaler.inverse_transform(y_test)
        plt.plot(actual_prices, label="Actual Price")
        plt.plot(predicted_prices, label="Predicted Price")
        plt.title(f"Actual vs Predicted Stock Price")
        plt.xlabel("Interval Count")
        plt.ylabel("Price (USD)")
        plt.legend()
        if showfig:
            plt.show()
        else:
            plt.savefig("outputs/model_eval.png")
            plt.clf()

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].numpy())
        y.append(data[i+seq_length].numpy())
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)


        
if __name__ == "__main__":
    # collecting and preprocessing data
    pandas_df = collect_data(["AAPL"], "2015-01-01", "2025-09-01", freq="1d")
    prices = torch.tensor(pandas_df.values, dtype=torch.float32)
    # turn raw data into sequences
    # X will be shape (batches, seq_len, input_dim)
    # y will be shape (batches, output_dim)
    seq_len = 60
    X, y = create_sequences(prices, seq_len)
    batches, seq_len, input_dim = X.shape
    # train test split ratio
    split = 0.8
    train_size = int(split * prices.shape[0])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    # scale data based on training data
    # scale across multiple dimensions to get min and max for each input_dim (for each stock)
    scaler = CustomMinMaxScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    scaled_y_train = scaler.transform(y_train)
    scaled_y_test = scaler.transform(y_test)

    # setup for training
    criterion = nn.MSELoss()
    lstm_model = StockLSTM(input_dim=input_dim, hidden_dim=64, output_dim=input_dim)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    trainer = ModelTrainer(
        criterion,
        lstm_model,
        optimizer,
        scaled_X_train,
        scaled_y_train,
    )
    # train
    trainer.train(num_epochs=100)
    trainer.plot_losses()
    # eval
    trainer.eval(scaled_X_test, scaled_y_test, scaler, False)

    transformer_model = StockTransformer()
    optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
    trainer = ModelTrainer(
        criterion,
        transformer_model,
        optimizer,
        scaled_X_train,
        scaled_y_train,
    )
    # trainer.train(num_epochs=30)
    # trainer.eval(scaled_X_test, scaled_y_test, scaler, True)

