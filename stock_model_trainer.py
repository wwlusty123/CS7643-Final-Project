import torch
import matplotlib.pyplot as plt

class ModelTrainer():
    def __init__(self, criterion, model, optimizer, X_train, y_train):
        self.criterion = criterion
        self.model = model
        self.optimizer = optimizer
        self.X_train = X_train
        self.y_train = y_train

    def train(self, num_epochs):
        for epoch in num_epochs:
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = self.criterion(output, self.y_train)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")
    
    def eval(self, X_test, y_test, scaler, showfig=False):
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(X_test).numpy()
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

        
if __name__ == "__main__":
    from fetch_stock_data import collect_data
    pandas_df = collect_data(["AAPL"], "2023-01-01", "2025-09-01", freq="1d")
    print(pandas_df.head())

