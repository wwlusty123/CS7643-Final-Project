import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def get_sp500_symbols():
    df = pd.read_csv("sp500.csv")
    return df["Symbol"].tolist()

def collect_data(symbols, start_date, end_date, freq="1d"):
    """
    Download historical closing-price data for one or more stock symbols.

    Parameters
    ----------
    symbols : str or list of str
        Ticker symbol(s) to download (e.g., "AAPL" or ["AAPL", "MSFT"]).
    start_date : str
        Start date for the data in "YYYY-MM-DD" format.
    end_date : str
        End date for the data in "YYYY-MM-DD" format.
    freq : str, optional
        Sampling interval supported by yfinance (e.g., "1d", "1h").
        Default is "1d".

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the closing prices for each symbol over
        the requested date range.
    """
    data = yf.download(symbols, start=start_date, end=end_date, interval=freq)
    data = data["Close"]
    return data

def get_eval_stocks():
    return ["AAPL", "JPM", "XOM", "BA", "UNH"]

def collect_eval_data():
    """
    Gather the evaluation dataset for the project.

    This function retrieves a fixed set of stock tickers over a stable
    date range. All model architectures will be evaluated on this dataset,
    which is intentionally kept separate from training and tuning to
    prevent data leakage.

    Returns
    -------
    pandas.DataFrame
        A dataframe of closing prices for the evaluation symbols.
    """
    symbols = get_eval_stocks()
    start_date, end_date = ("2024-11-01", "2025-10-31")
    eval_df = collect_data(symbols, start_date, end_date)
    return eval_df

if __name__ == "__main__":
    data = collect_data(["AAPL", "DE"], "2025-01-01", "2025-01-31", "1d")
    plt.plot(data.index, data["DE"])
    plt.plot(data.index, data["AAPL"])
    plt.title("DE and AAPL Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()