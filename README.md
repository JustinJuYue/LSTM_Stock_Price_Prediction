# Comprehensive Stock Analysis & Prediction Toolkit

This repository contains a powerful suite of Python tools for financial market analysis, visualization, and prediction. It is composed of two main parts: a versatile financial analysis toolkit and a sophisticated LSTM-based model for minute-level stock price prediction.

## 📜 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [File Structure](#-file-structure)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [How the LSTM Model Works](#-how-the-lstm-model-works)
- [Dependencies](#-dependencies)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Project Overview

This project provides a dual-functionality platform for traders, analysts, and data scientists:

1.  **Financial Analysis Toolkit (`LSTM_Mod...py`):** A collection of functions designed for in-depth stock analysis. It allows you to fetch historical data, create interactive plots (line charts, candlesticks), analyze correlations between multiple assets, plot technical indicators like moving averages, and visualize daily returns.

2.  **LSTM Prediction Model (`LSTM_Stock_...py`):** An advanced script that leverages a Long Short-Term Memory (LSTM) neural network to predict next-minute stock prices. It includes a unique "reinforcement" mechanism that automatically retrains the model with different learning rates if performance is suboptimal, ensuring the best possible prediction is chosen.

---

## ✨ Features

### Financial Analysis Toolkit
* **Interactive Stock Charting:** Plot line and candlestick charts for any stock ticker with custom date ranges and intervals (`Sg_Tracker_Plot`, `Sg_Tracker_Period`).
* **Correlation Analysis:** Calculate and visualize the correlation matrix between multiple stocks as an interactive heatmap (`Corr_Cal`).
* **Hierarchical Clustering:** Group stocks based on their price correlation using a dendrogram to identify clusters of related assets (`group_stocks_by_correlation`).
* **Moving Averages:** Plot single or multiple moving averages over a stock's closing price to identify trends (`movingAveragePlotter`).
* **Return Analysis:** Calculate and plot daily returns and their distribution in a histogram to understand volatility (`dailyReturnPlotter`, `plotDailyReturnHistogram`).
* **Ticker Scraping:** Fetch a comprehensive list of stock tickers from major indices like S&P 500, NASDAQ, FTSE, and more (`get_tickers`).

### LSTM Prediction Model
* **Minute-Level Prediction:** Predicts the closing price for the next minute using the previous 60 minutes of price and volume data.
* **Trading Signal Generation:** Classifies the prediction into a **Buy**, **Sell**, or **Hold** signal based on the expected price movement.
* **Smart Train/Test Split:** Uniquely uses the oldest day of data for testing and the most recent five days for training, simulating a real-world forecasting scenario.
* **Reinforcement Loop:** If the model's Mean Squared Error (MSE) on the test day is above a set threshold (e.g., 0.5), the script automatically re-runs the training with different learning rates to find a better-performing model.
* **Data-Driven:** Uses both **Price** and **Volume** as features for more robust predictions.

---

## 📁 File Structure

```
.
├── LSTM_Mod...py             # Main Financial Analysis Toolkit
├── LSTM_Stock_...py          # LSTM Prediction & Trading Signal Script
├── setup.py                  # (Optional) Project setup and package configuration
├── .gitignore                # Specifies files to be ignored by Git
└── README.md                 # This file
```

---

## ⚙️ Installation

To get this project up and running on your local machine, follow these steps.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Create a `requirements.txt` file with the content below and install it.
    ```
    # requirements.txt
    yfinance
    pandas
    numpy
    sympy
    matplotlib
    scikit-learn
    plotly
    dash
    yahoo_fin
    tensorflow
    ```
    Now, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage Examples

### Using the Financial Analysis Toolkit

You can import and use the functions from `LSTM_Mod...py` in another script or a Jupyter Notebook.

**Example 1: Plot Apple's stock price with moving averages**
```python
from LSTM_Mod... import movingAveragePlotter

movingAveragePlotter(
    ticker="AAPL",
    start_date="2024-01-01",
    end_date="2025-10-01",
    window=[20, 50]  # Plot 20-day and 50-day moving averages
)
```

**Example 2: Calculate and visualize the correlation between tech stocks**
```python
from LSTM_Mod... import Corr_Cal

tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'META']
correlation_matrix = Corr_Cal(
    Stocks=tech_stocks,
    start_input="2023,01,01"
)
print(correlation_matrix)
```

### Running the LSTM Prediction Model

The `LSTM_Stock_...py` script is designed to be run directly from the command line. It will execute the full pipeline for the specified ticker.

To run the prediction for Apple (`AAPL`):
```bash
python LSTM_Stock_...py
```
The script is pre-configured to run for "AAPL". You can change the ticker inside the `if __name__ == "__main__":` block at the end of the file.

The output will show:
- The training progress for each learning rate attempt.
- The final chosen learning rate and its corresponding Mean Squared Error (MSE).
- A summary of the generated trading signals (e.g., Buy: 35%, Hold: 40%, Sell: 25%).

---

## 🧠 How the LSTM Model Works

The prediction model follows a clear, logical pipeline designed for time-series forecasting:

1.  **Data Loading:** Downloads the last 6 days of minute-level data for a given stock.
2.  **Train/Test Split:** It cleverly separates the data:
    -   **Test Set:** The oldest day's data.
    -   **Training Set:** The most recent 5 days' data.
3.  **Preprocessing:** 'Close' and 'Volume' data are scaled to a range of [0, 1] using `MinMaxScaler`. This helps the neural network learn more effectively.
4.  **Windowing:** The data is transformed into "sliding windows." Each window consists of 60 minutes of data (`X`) and the price of the 61st minute as the label to be predicted (`y`).
5.  **Model Training:** An LSTM model is built and trained on the windowed training data. It learns patterns from the past 60 minutes to predict the next minute.
6.  **Prediction & Evaluation:** The trained model predicts prices for the test set. Its performance is measured by the Mean Squared Error (MSE) between the predicted and actual prices.
7.  **Reinforcement Loop:** If the initial MSE is higher than the `threshold` (e.g., 0.5), the entire process from training to evaluation is repeated with different learning rates (`[0.001, 0.005, 0.01]`). The model with the lowest MSE is ultimately selected.
8.  **Signal Generation:** For each prediction, a trading signal (Buy, Sell, or Hold) is generated by comparing the predicted price to the current price.

---

## 📦 Dependencies

- `yfinance`
- `pandas`
- `numpy`
- `sympy`
- `matplotlib`
- `scikit-learn`
- `plotly`
- `dash`
- `yahoo_fin`
- `tensorflow`

---

## ⚠️ Disclaimer

This script is intended for educational and informational purposes only.
The trading signals generated are based on a simplistic technical analysis strategy and should not be considered financial advice.
Trading involves risk, and you should do your own research or consult a licensed financial advisor before making any investment decisions.


---

## ⚖️ License

This project is open-source. Please add your preferred license (e.g., MIT, Apache 2.0) here.
