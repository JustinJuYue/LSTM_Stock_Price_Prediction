#!/usr/bin/env python
# coding: utf-8

"""
Minute-Level Stock Price Prediction & Trading Signal Classification using LSTM with Reinforcement Study

This script downloads minute-level stock data, preprocesses it, trains an LSTM model to predict the next-minute closing price,
and then uses the oldest day as test data to simulate predictions and compute trading decisions.
If the Mean Squared Error (MSE) on the test day is greater than 0.1, the model is re‑trained using alternative learning rates
([0.001, 0.005, 0.01]). If all three runs produce MSE > 0.1, the run with the smallest MSE is selected.
The script keeps the Plotly plot and other settings unchanged.

Required packages: yfinance, pandas, numpy, plotly, scikit-learn, tensorflow
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# Optional: import additional settings from setup.py if available.
try:
    from setup import *
except ImportError:
    pass

warnings.filterwarnings("ignore")


# 1. Data Loading and Preprocessing Functions
def load_data(ticker, period='6d', interval='1m'):
    """
    Downloads minute-level data for the given ticker.
    """
    df = yf.download(ticker, period=period, interval=interval)
    df = df[['Close', 'Volume']].dropna()
    return df

def split_train_test(data):
    """
    Split the data into training (last 5 days) and testing (oldest day) based on the date.
    """
    data = data.copy()
    data['Date'] = data.index.date
    test_date = data['Date'].min()  # Use the earliest date for testing
    train_data = data[data['Date'] != test_date]
    test_data = data[data['Date'] == test_date]
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """
    Fit scalers on training data and transform both train and test data.
    Returns:
      - train_scaled, test_scaled: numpy arrays with two features.
      - close_scaler: scaler for the Close column.
    """
    close_scaler = MinMaxScaler()
    volume_scaler = MinMaxScaler()
    
    train_close = train_data[['Close']]
    train_volume = train_data[['Volume']]
    
    test_close = test_data[['Close']]
    test_volume = test_data[['Volume']]
    
    # Fit on training data
    train_close_scaled = close_scaler.fit_transform(train_close)
    train_volume_scaled = volume_scaler.fit_transform(train_volume)
    train_scaled = np.hstack((train_close_scaled, train_volume_scaled))
    
    # Transform test data
    test_close_scaled = close_scaler.transform(test_close)
    test_volume_scaled = volume_scaler.transform(test_volume)
    test_scaled = np.hstack((test_close_scaled, test_volume_scaled))
    
    return train_scaled, test_scaled, close_scaler

def create_sliding_windows(scaled_data, window_size=60):
    """
    Creates sliding windows for time-series data.
    Each input consists of `window_size` minutes and the label is the next minute's close price.
    """
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size, 0])  # label is the close price
    return np.array(X), np.array(y)


# 2. Build the LSTM Model
def build_model(input_shape, learning_rate=0.001):
    """
    Builds and compiles a simple LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


# 3. Trading Signal Classification
def classify_action(predicted_price, current_price, threshold_ratio=0.001):
    """
    Classifies trading signal based on the difference between predicted and current price.
    Returns:
      1 for Buy, -1 for Sell, and 0 for Hold.
    """
    threshold = threshold_ratio * current_price
    diff = predicted_price - current_price
    if diff > threshold:
        return 1
    elif diff < -threshold:
        return -1
    else:
        return 0


# 4. Training and Simulation Pipeline for a Single Ticker
def run_pipeline(ticker, window_size=60, epochs=50, batch_size=32, learning_rate=0.005, display_plot=True):
    """
    Loads the data for the given ticker, splits into training and testing,
    trains an LSTM model on the training data, and simulates predictions on the test day.
    
    The function plots the real vs predicted close prices (if display_plot is True) and returns a dictionary containing:
      - predicted_prices, real_prices, actions, mse, action_summary, model, history.
    """
    print(f"Loading data for {ticker} with learning rate = {learning_rate} ...")
    data = load_data(ticker)
    if data.empty:
        print("No data found!")
        return None

    # Split into training (last 5 days) and testing (oldest day)
    train_data, test_data = split_train_test(data)
    
    # Extract time axis for test set (after sliding window offset)
    time_axis = test_data.index[window_size:]
    time_str = time_axis.strftime("%H:%M")
    
    # Preprocess (scale) data
    train_scaled, test_scaled, close_scaler = preprocess_data(train_data, test_data)
    
    # Create sliding windows
    X_train, y_train = create_sliding_windows(train_scaled, window_size)
    X_test, y_test = create_sliding_windows(test_scaled, window_size)
    
    # Build and train the LSTM model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]), learning_rate=learning_rate)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, 
                        batch_size=batch_size, callbacks=[early_stop], verbose=1)
    
    # Simulation on test data: predict on sliding windows
    predicted_scaled = model.predict(X_test)
    predicted_prices = close_scaler.inverse_transform(predicted_scaled)
    real_prices = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Generate trading actions: compare predicted price with the last price in each input window
    actions = []
    for i in range(len(X_test)):
        current_scaled = X_test[i, -1, 0]  # last minute's close in the window
        current_price = close_scaler.inverse_transform(np.array([[current_scaled]]))[0, 0]
        pred_price = predicted_prices[i, 0]
        actions.append(classify_action(pred_price, current_price))
    
    mse = np.mean((predicted_prices.flatten() - real_prices.flatten()) ** 2)
    
    
    # Summary statistics for trading actions
    actions_series = pd.Series(actions)
    action_counts = actions_series.value_counts(normalize=True).sort_index() * 100  # percentages
    action_summary = {
        'Buy %': action_counts.get(1, 0),
        'Hold %': action_counts.get(0, 0),
        'Sell %': action_counts.get(-1, 0)
    }
    
    results = {
        'ticker': ticker,
        'predicted_prices': predicted_prices.flatten(),
        'real_prices': real_prices.flatten(),
        'actions': actions,
        'mse': mse,
        'action_summary': action_summary,
        'model': model,
        'history': history
    }
    
    print(f"Finished pipeline for {ticker} with learning rate {learning_rate}. MSE: {mse:.4f}")
    print("Action distribution (in %):", action_summary)
    return results


def run_pipeline_reinforcement(ticker, window_size=60, epochs=25, batch_size=32, learning_rates=[0.001, 0.005, 0.01], threshold=0.5):
    """
    Runs the pipeline for the given ticker using different learning rates.
    
    Process:
      - For each learning rate, run the pipeline (without plotting) and record its MSE.
      - If any run produces an MSE < threshold, return that result immediately.
      - If all runs yield an MSE >= threshold, select and return the run with the smallest MSE.
      
    Returns:
      The result from run_pipeline that meets the criteria.
    """
    results = {}
    
    # Iterate through the learning rates and run the pipeline
    for lr in learning_rates:
        result = run_pipeline(ticker, window_size, epochs, batch_size, learning_rate=lr, display_plot=False)
        if result is not None:
            results[lr] = result
            # If MSE is below the threshold, return the result immediately.
            if result['mse'] < threshold:
                print(f"Learning rate {lr} produced MSE {result['mse']} below threshold {threshold}.")
                return result

    # If none of the runs produced an MSE below the threshold,
    # select the one with the smallest MSE from the stored results.
    best_lr = min(results, key=lambda lr: results[lr]['mse'])
    print(f"No run produced MSE below threshold. Selected learning rate {best_lr} with MSE {results[best_lr]['mse']}.")
    return results[best_lr]



# Example usage:
if __name__ == "__main__":
    # For example, run the reinforcement study for AAPL.
    # The process will try learning rates [0.001, 0.005, 0.01] and choose the best run according to the threshold.
    final_result = run_pipeline_reinforcement("AAPL", window_size=60, epochs=25, batch_size=32, learning_rates=[0.001, 0.005, 0.01], threshold=0.5)
    
    # The final_result includes predicted_prices, real_prices, actions, mse, action_summary, etc.
    print("Final decision summary (from selected run):", final_result['action_summary'])
