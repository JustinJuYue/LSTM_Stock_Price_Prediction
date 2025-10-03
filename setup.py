#!/usr/bin/env python
# coding: utf-8

# In[4]:

from setup import *
import yfinance as yf # Import real time data from Yahoo's database with lowest interval 60s
import time
from datetime import datetime
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import plotly.express as px
from datetime import datetime
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from yahoo_fin import stock_info as si
import builtins  # to ensure using the built-in list function


# In[ ]:


import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go

def Sg_Tracker_Plot(
        ticker_symbol: str, *, 
        start_input: str = "2022-11-22", 
        end_input: str = datetime.today().strftime('%Y-%m-%d'), 
        interval: str = '1d', 
        chart_type: str = 'line',
        plotData: str = 'Close') -> None:
    """
    Fetches historical stock data and displays an interactive price chart without returning a DataFrame.
    
    Parameters:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        start_input (str): Start date in 'yyyy-mm-dd' format
        end_input (str): End date in 'yyyy-mm-dd' format
        interval (str): Data interval (1m, 5m, 15m, 1h, 1d)
        chart_type (str): 'line' or 'candlestick' chart type
        plot_data (str): Price field to plot (Open, High, Low, Close)
        
    Raises:
        ValueError: For invalid inputs or data fetching failures
    """
    # Validate interval
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d']
    if interval not in valid_intervals:
        interval = '1d'

    # Validate and parse dates
    try:
        datetime.strptime(start_input, '%Y-%m-%d')
        datetime.strptime(end_input, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Invalid date format. Use 'yyyy-mm-dd'")

    # Fetch data
    ticker = yf.Ticker(ticker_symbol.upper())
    df = ticker.history(start=start_input, end=end_input, interval=interval, prepost=True)
    
    if df.empty:
        raise ValueError("No data fetched. Check inputs and try again.")
    
    # Prepare plot
    if chart_type == 'line':
        fig = go.Figure(go.Scatter(
            x=df.index,
            y=df[plotData],
            mode='lines',
            line=dict(width=1.5, color='royalblue'),
            name=ticker_symbol.upper()
        ))
    elif chart_type == 'candlestick':
        fig = go.Figure(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df[plotData],
            name='Price'
        ))
    else:
        raise ValueError("Invalid chart_type. Choose 'line' or 'candlestick'")

    # Update layout
    fig.update_layout(
        title=f"{ticker_symbol.upper()} {plotData} Price Movement",
        xaxis_title="Date/Time",
        yaxis_title=f"{plotData} Price (USD)",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False
    )
    
    if chart_type == 'candlestick':
        fig.update_layout(xaxis_rangeslider_visible=False)
    
    # Show plot
    fig.show()

    # Print latest price
    latest_price = df[plotData].iloc[-1]
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{current_time} - Latest {plotData} price: ${latest_price:.2f}")

#Sg_Tracker_Plot("AAPL",start_input='2010-01-01',end_input='2024-01-01',plotData='Volume')


# In[ ]:


def Sg_Tracker_Period(
        ticker_symbol: str, *,
        start_input: str = "2022-11-22",
        end_input: str = datetime.today().strftime('%Y-%m-%d'),
        interval: str = '1d',
        plot: bool = True,
        chart_type: str = 'line',
        plotData: str = 'Close') -> pd.DataFrame:
    """
    Fetches historical stock data using yfinance for a specified date range provided by the user,
    stores it in a sorted Pandas DataFrame, and optionally displays an interactive plot of the closing price movement.

    Parameters:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
        interval (str): The data interval (e.g., '1m', '5m', '15m'). Minimum is '1m'.
        start_input: Start Date (format: 'yyyy-mm-dd')
        end_input: End Date (format: 'yyyy-mm-dd')
        plot (bool): If True, displays an interactive plot of the closing price movement.
        chart_type (str): line or candlestick
        plotData (str): Open, High, Low, Close, Volume, Dividends, Stock Splits

    Returns:
        pd.DataFrame: DataFrame containing the historical stock data sorted by date/time.

    Raises:
        ValueError: If no data is fetched with the given parameters or if the date format is invalid.
    """

    # Validate the interval input
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d']
    if interval not in valid_intervals:
        interval = '1d'
    if interval.endswith("m"):
        try:
            minutes = int(interval[:-1])
            if minutes < 1:
                interval = '1m'
        except ValueError:
            interval = '1m'

    # Validate date format
    try:
        datetime.strptime(start_input, '%Y-%m-%d')
        datetime.strptime(end_input, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Invalid date format. Please enter dates as 'yyyy-mm-dd'")

    # Fetch historical data using start and end dates
    ticker = yf.Ticker(ticker_symbol.upper())
    df = ticker.history(start=start_input, end=end_input, interval=interval, prepost=True)
    if df.empty:
        raise ValueError("No data fetched. Check the ticker symbol, interval, or date range.")
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)

    # Identify the column holding the date/time information (usually the first column)
    date_col = df.columns[0]

    latest_price = df['Close'].iloc[-1]
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{current_time} - Latest closing price of {ticker_symbol.upper()}: ${latest_price:.2f}")

    if plot:
        if chart_type == 'line':
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[plotData],
                mode='lines',
                marker=dict(size=8, color='blue'),
                line=dict(width=2),
                name=ticker_symbol.upper(),
                hovertemplate='<b>Date:</b> %{x}<br><b>Close:</b> %{y:.2f}<extra></extra>',
            ))
            fig.update_layout(
                title=f"{ticker_symbol.upper()} Stock Price Movement",
                xaxis_title="Date/Time",
                yaxis_title=plotData + " value",
                template="plotly_white",
                hovermode="closest"
            )
        elif chart_type == 'candlestick':
            fig = go.Figure(data=[go.Candlestick(
                x=df[date_col],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df[plotData],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            fig.update_layout(
                title=f"{ticker_symbol.upper()} Candlestick Chart",
                xaxis_title="Date/Time",
                yaxis_title="Price (USD)",
                template="plotly_white",
                hovermode="closest",
                xaxis_rangeslider_visible=False
            )
        else:
            raise ValueError("Invalid chart_type. Choose 'line' or 'candlestick'.")
        fig.show()

    return df

# Sg_Tracker_Period("AAPL",plotData='Volume')


# In[7]:


def Corr_Cal(Stocks, *, start_input="2010,11,22",end_date = datetime.today().strftime('%Y-%m-%d'), interval: str = '1d'):
    """
    Calculate the correlation between stocks over a specified time period and display a correlation heatmap.

    Parameters:
        Stocks (list): A list of stock ticker symbols (e.g., ['AAPL', 'META']).
        start_input (str): Start date in format "yyyy,mm,dd".
        interval (str): The data interval (e.g., '1m', '5m', '15m', '1d'). Minimum is '1m'.

    Returns:
        pd.DataFrame: The correlation matrix of the stocks' closing prices during the specified period.
    """
    # Parse start date
    try:
        start_list = [int(x.strip()) for x in start_input.split(",")]
        start_date = f"{start_list[0]:04d}-{start_list[1]:02d}-{start_list[2]:02d}"
    except Exception as e:
        raise ValueError("Invalid start date format. Please enter the date as yyyy,mm,dd") from e
    

    # Validate interval
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '1h', '1d']
    if interval not in valid_intervals:
        interval = '1d'
    
    # Download historical stock data
    df = yf.download(tickers=Stocks, start=start_date, end=end_date, interval=interval, group_by='ticker')
    if df.empty:
        raise ValueError("No data fetched. Check the ticker symbols, interval, or date range.")
    
    # Extract closing prices
    close_data = pd.DataFrame()
    for stock in Stocks:
        try:
            close_data[stock] = df[stock]['Close']
        except Exception as e:
            raise ValueError(f"Error extracting closing price for {stock}") from e

    # Compute correlation matrix
    corr = close_data.corr()

    # Generate correlation heatmap using Plotly
    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix of Stocks")
    fig.update_layout(xaxis_title="Stocks", yaxis_title="Stocks")
    fig.show()
    
    return corr



# In[8]:


def get_tickers(Output_csvfile=False):
    """
    Attempt to fetch tickers from various markets; if a function is not available, default to an empty list.

    Parameters:
        Output_csvfile (bool): If True, generates a file called 'all_stock_tickers.csv' in the current directory.
                               Default is False.
    Returns:
        A list containing the unique tickers (with empty values removed).
    """
    try:
        dow_tickers = si.tickers_dow()
    except Exception:
        dow_tickers = []

    try:
        ftse100_tickers = si.tickers_ftse100()  # UK FTSE 100 tickers
    except Exception:
        ftse100_tickers = []

    try:
        ftse250_tickers = si.tickers_ftse250()  # UK FTSE 250 tickers
    except Exception:
        ftse250_tickers = []

    try:
        ibovespa_tickers = si.tickers_ibovespa()  # Brazil's IBOVESPA tickers
    except Exception:
        ibovespa_tickers = []

    try:
        nasdaq_tickers = si.tickers_nasdaq()
    except Exception:
        nasdaq_tickers = []

    try:
        nifty50_tickers = si.tickers_nifty50()  # India's NIFTY 50 tickers
    except Exception:
        nifty50_tickers = []

    try:
        niftybank_tickers = si.tickers_niftybank()  # India's NIFTY Bank tickers
    except Exception:
        niftybank_tickers = []

    try:
        sp500_tickers = si.tickers_sp500()
    except Exception:
        sp500_tickers = []

    try:
        other_tickers = si.tickers_other()
    except Exception:
        other_tickers = []

    # Combine all tickers into a set to remove duplicates
    all_tickers_set = set(
        dow_tickers +
        ftse100_tickers +
        ftse250_tickers +
        ibovespa_tickers +
        nasdaq_tickers +
        nifty50_tickers +
        niftybank_tickers +
        sp500_tickers +
        other_tickers
    )

    # Convert the set back to a list using builtins.list to avoid name conflicts
    unique_tickers_list = builtins.list(all_tickers_set)

    # Drop any empty strings from the list
    unique_tickers_list = [ticker for ticker in unique_tickers_list if ticker]

    # Print total unique tickers
    print(f"Total unique tickers: {len(unique_tickers_list)}")

    # Save to CSV if requested
    if Output_csvfile:
        tickers_df = pd.DataFrame(unique_tickers_list, columns=['Ticker'])
        tickers_df.to_csv('all_stock_tickers.csv', index=False)
        print("CSV file 'all_stock_tickers.csv' has been created.")

    return unique_tickers_list


# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from datetime import datetime

def group_stocks_by_correlation(Stocks, *, start_input="2010-11-22", end_input=datetime.today().strftime('%Y-%m-%d'), interval='1d', HCD = False):
    """
    Fetches historical stock data for a list of stocks, computes the correlation matrix of their closing prices,
    performs hierarchical clustering to group stocks by their historical price correlation, and displays a 
    correlation heatmap and dendrogram.

    Parameters:
        Stocks (list): List of stock ticker symbols (e.g., ['AAPL', 'META', 'GOOG']).
        start_input (str): Start date in format "yyyy-mm-dd". (Default: "2010-11-22")
        end_input (str): End date in format "yyyy-mm-dd". (Default: current date in "yyyy-mm-dd" format)
        interval (str): Data interval (e.g., '1m', '1d', etc.). Default is '1d'.

    Returns:
        pd.DataFrame: The correlation matrix of the stocks' closing prices.
    """
    # Parse start date (expects format yyyy-mm-dd)
    try:
        start_date_obj = datetime.strptime(start_input, "%Y-%m-%d")
        start_date = start_date_obj.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError("Invalid start date format. Please enter the date as yyyy-mm-dd") from e

    # Parse end date (expects format yyyy-mm-dd)
    try:
        end_date_obj = datetime.strptime(end_input, "%Y-%m-%d")
        end_date = end_date_obj.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError("Invalid end date format. Please enter the date as yyyy-mm-dd") from e

    # Download historical data for each stock
    close_data = pd.DataFrame()
    for stock in Stocks:
        try:
            data = yf.Ticker(stock).history(start=start_date, end=end_date, interval=interval, prepost=True)
            if data.empty:
                print(f"No data for {stock}")
            else:
                close_data[stock] = data['Close']
        except Exception as e:
            print(f"Error retrieving data for {stock}: {e}")

    if close_data.empty:
        raise ValueError("No data fetched for any of the provided stocks.")

    # Handle missing values by forward-filling (you can adjust this strategy as needed)
    close_data.fillna(method='ffill', inplace=True)
    # Optionally, drop any stocks with insufficient data
    close_data.dropna(axis=1, inplace=True)

    # Compute correlation matrix
    corr = close_data.corr()

    # Plot correlation heatmap using Plotly
    fig_heat = px.imshow(corr, text_auto=True, title="Correlation Matrix of Stocks")
    fig_heat.update_layout(xaxis_title="Stocks", yaxis_title="Stocks")
    fig_heat.show()

    if HCD:
        # Compute distance matrix (using 1 - correlation as distance)
        distance_matrix = 1 - corr

        # Perform hierarchical clustering using complete linkage
        linkage = sch.linkage(distance_matrix, method='complete')

        # Plot dendrogram using Matplotlib
        plt.figure(figsize=(12, 8))
        dendro = sch.dendrogram(linkage, labels=corr.columns, leaf_rotation=90)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Stock")
        plt.ylabel("Distance (1 - Correlation)")

        plt.tight_layout()
        plt.show()

    return corr




# In[13]:


import matplotlib.pyplot as plt
import datetime as dt

def movingAveragePlotter(ticker: str, start_date: str = None, end_date: str = None, window=None, interval="1d"):
    """
    Fetch historical stock data using Sg_Tracker_Period and plot the closing price along with 
    one or multiple moving averages.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - start_date (str): Start date in "YYYY-MM-DD" format (defaults to one year ago from today).
    - end_date (str): End date in "YYYY-MM-DD" format (defaults to today's date).
    - window (int or list of int): Moving average window size(s). Defaults to [5, 10, 20, 50].
    """
    # Set default end_date to today if not provided
    if end_date is None:
        end_date_obj = dt.date.today()
        end_date = end_date_obj.strftime("%Y-%m-%d")
    else:
        end_date_obj = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Set default start_date to one year ago if not provided
    if start_date is None:
        start_date_obj = end_date_obj - dt.timedelta(days=365)
        start_date = start_date_obj.strftime("%Y-%m-%d")
    else:
        start_date_obj = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    
    # Directly use the proper format "YYYY-MM-DD"
    start_input = start_date
    end_input = end_date
    
    # Set default moving average windows if not provided
    if window is None:
        window = [5, 10, 20, 50]
    elif isinstance(window, int):
        window = [window]
    
    # Attempt to retrieve historical data using Sg_Tracker_Period
    try:
        df = Sg_Tracker_Period(ticker, start_input=start_input, end_input=end_input, plot=False, interval=interval)
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return

    # Check if data is valid
    if df is None or df.empty or 'Close' not in df.columns:
        print("No data available for the given ticker or date range.")
        return
    
    # Compute moving averages for each window
    for w in window:
        df[f"MA_{w}"] = df['Close'].rolling(window=w).mean()

    # Plot the closing price and moving averages
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label='Closing Price', linewidth=2)
    
    for w in window:
        plt.plot(df.index, df[f"MA_{w}"], label=f'{w}-Day Moving Average', linestyle='--')
    
    plt.title(f"{ticker.upper()} Closing Prices with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()


# In[19]:


import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np

# --- Updated dailyReturnPlotter Function ---
def dailyReturnPlotter(ticker: str, start_date: str = None, end_date: str = None):
    """
    Fetch historical stock data using Sg_Tracker_Period and plot the daily returns of the stock.
    
    Parameters:
    - ticker (str): Stock ticker symbol.
    - start_date (str): Start date in "yyyy-mm-dd" format (defaults to one year ago from today).
    - end_date (str): End date in "yyyy-mm-dd" format (defaults to today's date).
    """
    # ---------------------------
    # 1. HANDLE DATE PARAMETERS
    # ---------------------------
    # If no end_date is provided, default to today's date
    if end_date is None:
        end_date_obj = dt.date.today()
        end_date = f"{end_date_obj.year}-{end_date_obj.month:02d}-{end_date_obj.day:02d}"
    else:
        end_date_obj = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # If no start_date is provided, default to 1 year before end_date
    if start_date is None:
        start_date_obj = end_date_obj - dt.timedelta(days=365)
        start_date = f"{start_date_obj.year}-{start_date_obj.month:02d}-{start_date_obj.day:02d}"
    else:
        start_date_obj = dt.datetime.strptime(start_date, "%Y-%m-%d").date()

    # ---------------------------
    # 2. FETCH DATA
    # ---------------------------
    try:
        df = Sg_Tracker_Period(ticker, start_input=start_date, end_input=end_date, plot=False)
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return

    if df is None or df.empty or 'Close' not in df.columns:
        print("No data available for the given ticker or date range.")
        return

    # ---------------------------
    # 3. ENSURE A DATETIME INDEX
    # ---------------------------
    # If the DataFrame has a 'Date' column, convert it to datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
    else:
        # Otherwise, try to convert the existing index to datetime
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
        except Exception as e:
            print(f"Could not parse the index as datetime: {e}")
            return

    df.sort_index(inplace=True)

    # ---------------------------
    # 4. CALCULATE DAILY RETURNS
    # ---------------------------
    df['DailyReturn'] = df['Close'].pct_change()
    avg_daily_return = df['DailyReturn'].mean()
    print(f"Average Daily Return for {ticker.upper()}: {avg_daily_return:.4f}")

    # ---------------------------
    # 5. PLOT THE DAILY RETURNS
    # ---------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['DailyReturn'], label='Daily Return', color='blue', linewidth=1)
    plt.axhline(avg_daily_return, color='red', linestyle='--', label=f'Avg Daily Return ({avg_daily_return:.4f})')
    plt.title(f"Daily Returns for {ticker.upper()}")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# In[20]:


import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

def plotDailyReturnHistogram(ticker: str, start_date: str = None, end_date: str = None, interval: str = '1d'):
    """
    Fetch historical stock data using Sg_Tracker_Period for one company,
    calculate the daily returns, and plot a histogram of these returns.
    
    Parameters:
    - ticker (str): Stock ticker symbol.
    - start_date (str): Start date in "yyyy-mm-dd" format (defaults to one year ago from today).
    - end_date (str): End date in "yyyy-mm-dd" format (defaults to today's date).
    - interval (str): Data interval (default is '1d').
    """
    # Set default end_date to today's date if not provided.
    if end_date is None:
        end_date_obj = dt.date.today()
        end_date = end_date_obj.strftime("%Y-%m-%d")
    else:
        end_date_obj = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Set default start_date to one year ago if not provided.
    if start_date is None:
        start_date_obj = end_date_obj - dt.timedelta(days=365)
        start_date = start_date_obj.strftime("%Y-%m-%d")
    else:
        start_date_obj = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    
    # Retrieve historical data using Sg_Tracker_Period.
    try:
        df = Sg_Tracker_Period(ticker, start_input=start_date, end_input=end_date, interval=interval, plot=False)
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return
    
    if df is None or df.empty or 'Close' not in df.columns:
        print("No data available for the given ticker or date range.")
        return
    
    # Convert the index to a proper DatetimeIndex.
    try:
        df.index = pd.to_datetime(df.index, errors='coerce')
    except Exception as e:
        print(f"Error converting index to datetime: {e}")
        return
    
    df.sort_index(inplace=True)
    
    # Calculate daily returns using the percentage change of 'Close'.
    df['DailyReturn'] = df['Close'].pct_change()
    avg_return = df['DailyReturn'].mean()
    print(f"Average Daily Return for {ticker.upper()}: {avg_return:.4f}")
    
    # Plot histogram of daily returns.
    plt.figure(figsize=(12, 9))
    plt.hist(df['DailyReturn'].dropna(), bins=50, edgecolor='black')
    plt.axvline(avg_return, color='red', linestyle='--', linewidth=2, label=f'Avg Daily Return ({avg_return:.4f})')
    plt.xlabel("Daily Return")
    plt.ylabel("Counts")
    plt.title(f"{ticker.upper()} Daily Return Histogram")
    plt.legend()
    plt.tight_layout()
    plt.show()


