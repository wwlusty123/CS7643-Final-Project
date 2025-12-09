from torch.nn.modules.loss import _Loss

from fetch_stock_data import get_eval_stocks
from gru_scaler import BaseScaler, StandardScaler
from stock_model_trainer import train_eval
from stock_predictor_models import StockGRU
import yfinance as yf
import pandas as pd
import fetch_stock_data as data
import stock_model_trainer as trainer
import stock_predictor_models as models
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import numpy as np

def gru_driver():
    scaler = StandardScaler()
    X_train, X_test, X_holdout, y_train, y_test, y_holdout = get_train_test_holdout_data(scaler=scaler)

    input_dim = 5  # 5 stocks
    hidden_dim = 32
    output_dim = 5  # predict all 5 stocks
    num_layers = 2

    model = StockGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.0,
        num_layers=num_layers
    )

    train_losses, test_losses, preds_test, metrics = train_and_test_gru_model(
        model, X_train, y_train, X_test, y_test, epochs=50, criterion=nn.SmoothL1Loss(beta=0.1)
    )
    print(pd.DataFrame.from_records([metrics]))
    plot_test_predictions_subplots(
        preds_test, y_test, scaler=scaler, stock_names=["AAPL", "JPM", "XOM", "BA", "UNH"]
    )

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.legend()
    plt.title("GRU Training Curve")
    plt.show()


def get_train_test_holdout_data(scaler: BaseScaler, seq_len=30):
    # Load evaluation stock dataset
    df_all = pd.read_csv("~/CS7643-Final-Project/sp500_dataset.csv").set_index("Date")
    df_all = df_all[["AAPL", "MSFT", "AMZN", "GOOG", "JPM"]]            # keep only 5 eval stocks
    df_all = df_all.dropna(axis=1, how="any")     # ensure no missing columns

    # Convert to returns
    df_all = df_all.pct_change().dropna()

    # Convert to torch
    data_all = torch.tensor(df_all.values, dtype=torch.float32)

    # Split by date BEFORE scaling
    idx = df_all.index
    train_mask   = idx < "2024-01-01"
    test_mask    = (idx >= "2024-01-01") & (idx <= "2024-10-31")
    holdout_mask = (idx >= "2024-11-01") & (idx <= "2025-10-31")

    train_data   = data_all[train_mask]
    test_data    = data_all[test_mask]
    holdout_data = data_all[holdout_mask]

    # ----------------------------
    # Apply SCALER (NEW)
    # ----------------------------
    scaler.fit(train_data)              # fit only on training

    train_scaled   = scaler.transform(train_data)
    test_scaled    = scaler.transform(test_data)
    holdout_scaled = scaler.transform(holdout_data)

    # ----------------------------
    # Create sequences (X, y)
    # ----------------------------
    X_train, y_train     = trainer.create_sequences(train_scaled, seq_len)
    X_test, y_test       = trainer.create_sequences(test_scaled, seq_len)
    X_holdout, y_holdout = trainer.create_sequences(holdout_scaled, seq_len)

    return (
        X_train, X_test, X_holdout,
        y_train, y_test, y_holdout,
    )


def train_and_test_gru_model(model, X_train, y_train, X_test, y_test, criterion: _Loss = nn.MSELoss(),
                             lr=1e-3, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses  = []

    pred_test_final = None   # final predictions

    for epoch in range(epochs):
        # ============================
        # TRAIN
        # ============================
        model.train()
        optimizer.zero_grad()

        pred_train = model(X_train)
        loss = criterion(pred_train, y_train)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # ============================
        # TEST
        # ============================
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test)
            test_loss = criterion(pred_test, y_test)

        test_losses.append(test_loss.item())
        pred_test_final = pred_test.detach().clone()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss = {loss.item():.6f} | "
                  f"Test Loss = {test_loss.item():.6f}")

    # ============================================================
    # Compute evaluation metrics using final predictions
    # ============================================================
    preds = pred_test_final.cpu().numpy()
    truth = y_test.cpu().numpy()

    # ---- 1. MSE per stock ----
    mse_per_stock = ((preds - truth) ** 2).mean(axis=0)

    # ---- 2. SMAPE per stock ----
    smape_per_stock = (
        200 * np.abs(preds - truth) / (np.abs(preds) + np.abs(truth) + 1e-8)
    ).mean(axis=0)

    # ---- 3. MAE per stock ----
    mae_per_stock = (np.abs(preds - truth)).mean(axis=0)

    # ---- 4. Directional Accuracy ----
    directional_accuracy_per_stock =  ((preds * truth) > 0).mean(axis=0)
    average_directional_accuracy = directional_accuracy_per_stock.mean()


    metrics = {
        "average_smape": smape_per_stock.mean(),
        'average_mse': mse_per_stock.mean(),
        "average mae": mae_per_stock.mean(),
        "average directional accuracy": average_directional_accuracy.mean()
    }

    return train_losses, test_losses, pred_test_final, metrics

def plot_test_predictions_subplots(preds, y_true, scaler=None, stock_names=None):
    if scaler is not None:
        preds = scaler.inverse_transform(preds)
        y_true = scaler.inverse_transform(y_true)

    preds = preds.cpu().numpy()
    y_true = y_true.cpu().numpy()

    num_stocks = preds.shape[1]
    if stock_names is None:
        stock_names = [f"Stock {i}" for i in range(num_stocks)]

    plt.figure(figsize=(20, 3 * num_stocks))

    for i in range(num_stocks):
        plt.subplot(num_stocks, 1, i + 1)
        plt.plot(y_true[:, i], label="True", linewidth=2)
        plt.plot(preds[:, i], label="Pred", linewidth=2)
        plt.title(stock_names[i])
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    gru_driver()