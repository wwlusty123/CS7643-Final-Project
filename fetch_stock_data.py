import yfinance as yf
import matplotlib.pyplot as plt

def collect_data(symbols, start_date, end_date, freq):
    data = yf.download(symbols, start=start_date, end=end_date, interval=freq)
    data = data["Close"]
    return data

if __name__ == "__main__":
    data = collect_data(["AAPL", "DE"], "2025-01-01", "2025-01-31", "1d")
    plt.plot(data.index, data["DE"])
    plt.plot(data.index, data["AAPL"])
    plt.title("DE and AAPL Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()