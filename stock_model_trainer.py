import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from fetch_stock_data import collect_data, collect_eval_data, get_eval_stocks
import pandas as pd
from stock_predictor_models import StockLSTM, CustomMinMaxScaler, StockTransformer, StockCNN
import torch.optim as optim

class ModelTrainer():
    def __init__(
        self, 
        criterion, 
        model, 
        optimizer, 
        X_train, 
        y_train, 
        X_test=None, 
        y_test=None,
        scheduler=None,
    ):
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.losses = []
        self.test_losses = []
        self.has_test_data = self.X_test is not None and self.y_test is not None
        self.scheduler = scheduler


    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = self.criterion(output, self.y_train)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.losses.append(loss.item())
            if self.has_test_data:
                self.model.eval()
                with torch.no_grad():
                    pred_test = self.model(self.X_test)
                test_loss = self.criterion(pred_test, self.y_test)
                self.test_losses.append(test_loss.item())
            if (epoch + 1) % 5 == 0:
                print_str = f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.6f}"
                if self.has_test_data:
                    print_str += f", Test Loss: {test_loss.item():.6f}"
                print(print_str)

    def plot_losses(self, fig_path="model_losses.png", showfig=False):
        if len(self.losses) == 0:
            raise Exception("Must train a model prior to plotting losses")
        plt.plot(self.losses, label="train")
        if len(self.test_losses) > 0:
            plt.plot(self.test_losses, label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Model Training Curve")
        plt.legend()
        if showfig:
            plt.show()
        else:
            plt.savefig(f"outputs/{fig_path}")
            plt.clf()
    def eval(self, X_test, y_test, scaler, showfig=False, fig_path="model_eval.png"):
        """
        Evaluate the model on a given set of pre-scaled, preprocessed (that is, broken into sequences) data.
        """
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
            plt.savefig(f"outputs/{fig_path}")
            plt.clf()

class StockModelEvaluator():
    def eval(self, stock_preds):
        """
        Evaluate model predictions against the held-out evaluation dataset.

        This method compares the predicted stock prices in `stock_preds` to the
        project’s fixed evaluation set and computes error metrics for each stock 
        as well as an overall summary:

            1. Mean Squared Error (MSE) per stock
            2. Symmetric Mean Absolute Percentage Error (SMAPE) per stock
            3. Average SMAPE across all stocks

        Parameters
        ----------
        stock_preds : pandas.DataFrame
            DataFrame containing model predictions with columns 
            ["AAPL", "JPM", "XOM", "BA", "UNH"], covering the date range 
            "2024-11-01" to "2025-10-31". Index should align with the 
            evaluation set’s dates.

        Returns
        -------
        dict
            Dictionary with the following keys:
                - "per_stock_mse" : dict
                    Mapping each stock ticker to its Mean Squared Error.
                - "per_stock_smape" : dict
                    Mapping each stock ticker to its Symmetric Mean Absolute 
                    Percentage Error.
                - "average_smape" : float
                    Average SMAPE across all stocks.
        """
        eval_stocks = get_eval_stocks()
        if not np.all(stock_preds.columns.values == eval_stocks):
            raise ValueError(f"stock_preds should contain the correct stocks in the right order: {eval_stocks}")
        # rename the predicted columns: "AAPL" -> "AAPL_pred"   
        df_preds = stock_preds.rename(columns=dict(zip(eval_stocks, [f"{stock}_pred" for stock in eval_stocks]))) 
        df_truth = collect_eval_data()
        num_days_before, _ = df_truth.shape
        df_merged = df_truth.join(df_preds, how="inner")
        num_days_after, _ = df_merged.shape
        if num_days_before != num_days_after:
            raise ValueError(f"Lost some days in the evaluation period when joining to predictions. Rows before -> Rows After = {num_days_before} -> {num_days_after}")
        # add squared errors, symmetric absolute percent errors to the dataframe
        df_errors = df_merged.copy()
        for stock in eval_stocks:
            df_errors[f"{stock}_SE"] = (df_errors[f"{stock}_pred"] - df_errors[stock]) ** 2
            df_errors[f"{stock}_SAPE"] = 2 * 100 * (df_errors[f"{stock}_pred"] - df_errors[stock]).abs() / (df_errors[f"{stock}_pred"] + df_errors[stock])
        # averaging over dates
        df_MSEs = df_errors.mean(axis=0)[[f"{stock}_SE" for stock in eval_stocks]]
        df_MSEs.index = df_MSEs.index.str.replace("_SE", "_MSE", regex=False)
        dict_MSEs = df_MSEs.to_dict()
        df_SMAPEs = df_errors.mean(axis=0)[[f"{stock}_SAPE" for stock in eval_stocks]]
        df_SMAPEs.index = df_MSEs.index.str.replace("_SAPE", "_SMAPE", regex=False)
        dict_SMAPEs = df_SMAPEs.to_dict()
        # average over stocks
        avg_SMAPE = df_SMAPEs.mean()
        return {
            "per_stock_mse": dict_MSEs, 
            "per_stock_smape": dict_SMAPEs, 
            "average_smape": avg_SMAPE
        } 
        

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].numpy())
        y.append(data[i+seq_length].numpy())
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)

def train_eval(
    model_class, 
    kwargs=None, 
    seq: int = 60, 
    epochs: int = 50, 
    lr: float = 1e-3, 
    start: str = "2015-01-01", 
    end: str = "2024-10-31",
    train_fig_path="model_losses.png",
    eval_fig_path="model_eval.png",
):
    if kwargs is None:
        kwargs = {}
    
    # 1) symbols and data for training + eval
    eval_df = collect_eval_data()
    eval_last = eval_df.index[-1]
    buffer = (eval_last + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    symbols = get_eval_stocks()
    full_df = collect_data(symbols, start, buffer, freq="1d")
    full_df = full_df.dropna()
    prices_all = torch.tensor(full_df.values, dtype=torch.float32)
    dates_all = np.array(full_df.index)

    # 2) make sequences: x_all, y_all, and target_dates
    X_all, y_all = create_sequences(prices_all, seq)
    windows, _, input = X_all.shape
    target = dates_all[seq:]

    # 3) select training windows by date
    train = target <= np.datetime64(end)
    X_train = X_all[train]
    y_train = y_all[train]
    
    # 4) scale trainin data
    scaler = CustomMinMaxScaler()
    X = scaler.fit_transform(X_train)
    y = scaler.transform(y_train)
    X = X.to("cpu")
    y = y.to("cpu")

    # 5) Build and train the model
    mk = dict(kwargs)
    mk.setdefault("input_dim", input)
    mk.setdefault("output_dim", input)
    model = model_class(**mk).to("cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = ModelTrainer(
        criterion=criterion,
        model=model,
        optimizer=optimizer,
        X_train=X,
        y_train=y
    )
    trainer.train(num_epochs=epochs)
    trainer.plot_losses(fig_path=train_fig_path)
    # plot predictions from in-sample
    trainer.eval(X, y, scaler, fig_path=eval_fig_path)

    # 6) Use the trained model to predict all target data
    X_all_scaled = scaler.transform(X_all).to("cpu")
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_all_scaled)

    preds = scaler.inverse_transform(preds_scaled.cpu())
    preds_df_all = pd.DataFrame(
        preds.numpy(),
        index=target,
        columns=symbols
    )

    # 7) Restrict to the official eval period + column order
    eval_df = eval_df[get_eval_stocks()]
    preds_df = preds_df_all.loc[eval_df.index]
    preds_df = preds_df[get_eval_stocks()]

    # 8) Run the official evaluator
    evaluator = StockModelEvaluator()
    results = evaluator.eval(preds_df)

    return preds_df, results, model, scaler


        
if __name__ == "__main__":
    # collecting and preprocessing data
    pandas_df = collect_data(["AAPL", "GOOG"], "2015-01-01", "2024-09-01", freq="1d")
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
    cnn = StockCNN(input=input_dim, hidden=32, output=input_dim)
    optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
    trainer = ModelTrainer(
        criterion,
        cnn,
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

