# 📊 LSTM Stock Price Prediction & Analysis Toolkit

This project uses a **Long Short-Term Memory (LSTM)** neural network to predict **minute-level stock prices** and generate **trading signals**. It features a unique **"reinforcement study"** mechanism that automatically retrains the model with different learning rates to improve performance if the initial model underperforms.

Additionally, the repository includes a comprehensive **analysis toolkit** (`setup.py`) with powerful utility functions for advanced stock data visualization and insights.

---

## 🚀 Key Features

- **📈 Minute-Level Price Prediction**  
  Trains an LSTM model on the last **5 days** of minute-level stock data to predict the **next minute's closing price**.

- **⚙️ Automated Model Optimization**  
  If the model's **Mean Squared Error (MSE)** is above a defined threshold, it automatically retries training with different **learning rates**.

- **📉 Trading Signal Generation**  
  Generates **Buy (1), Sell (-1), or Hold (0)** signals based on predicted price movements.

- **🧰 Comprehensive Analysis Toolkit (`setup.py`)**  
  - Plot **interactive price charts** (line or candlestick)  
  - Generate **correlation heatmaps** between multiple stocks  
  - Visualize **moving averages** and **return distributions**  
  - Perform **hierarchical clustering** on stocks by correlation

---

## 📂 File Structure
```bash
LSTM_Stock_Price_Prediction/
│
├── pycache/ # Python cache files
├── .gitignore # Git ignore file
├── LSTM_Model.py # Core script for LSTM model training & prediction
├── README.md # Project documentation (this file)
└── setup.py # Stock analysis & visualization toolkit
```
---
## 🛠️ Technologies Used

- **Python**: Core language
- **TensorFlow & Keras**: LSTM model building and training
- **yfinance & yahoo_fin**: For historical and live stock data
- **Pandas & NumPy**: Data manipulation and processing
- **Scikit-learn**: Preprocessing with `MinMaxScaler`
- **Matplotlib & Plotly**: Data visualization (static and interactive)
- **SciPy & Dash**: Advanced analytics and dashboarding

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/LSTM_Stock_Price_Prediction.git
cd LSTM_Stock_Price_Prediction
```
### 2. Install Required Packages
Create a 'requirements.txt' file with the following contents:
```bash
tensorflow
yfinance
yahoo_fin
pandas
numpy
scikit-learn
plotly
matplotlib
scipy
dash
```
Then, run:
```bash
pip install -r requirements.txt
```

# LSTM Stock Prediction & Analysis Toolkit

## ▶️ How to Use

### 1. Run the LSTM Prediction Model

Edit the ticker symbol in the `if __name__ == "__main__"` block at the bottom of `LSTM_Model.py`:


```bash
`# In LSTM_Model.py
if __name__ == "__main__":
    # Change "TSLA" to your desired stock symbol
    final_result = run_pipeline_reinforcement("TSLA")`
```
Then run the script:

```bash

`python LSTM_Model.py`
```
**Outputs:**

- Training progress with loss and Mean Squared Error (MSE)
- Final MSE after model selection
- Count of Buy, Sell, and Hold signals

---

### 2. Use the Analysis Toolkit (setup.py)

Import and use the helper functions in a Jupyter Notebook or a separate script.

**Example: Plot Historical Stock Price**

```bash

`from setup import Sg_Tracker_Plot

# Plot Apple's closing price from 2020 to 2023
Sg_Tracker_Plot(
    "AAPL",
    start_input='2020-01-01',
    end_input='2023-01-01',
    chart_type='line',
    plotData='Close'
)`
```

**Example: Correlation Between Tech Stocks**

```bash

`from setup import Corr_Cal

tech_stocks = ['AAPL', 'GOOGL', 'MSFT', 'META']

# Calculate and display the correlation matrix
correlation_matrix = Corr_Cal(tech_stocks, start_input="2022,1,1")
print(correlation_matrix)`
```
---

## 🧠 How It Works

1. **📥 Data Collection**: Uses `yfinance` to download the last 6 days of minute-level data. The most recent 5 days are used for training, and the oldest day is reserved for testing.
2. **⚙️ Preprocessing**: Scales **Close** price and **Volume** between 0 and 1 using `MinMaxScaler`.
3. **🧩 Sliding Window Input**: Creates 60-minute sequences as input. The model's task is to predict the 61st-minute closing price.
4. **🧠 LSTM Model Training**: Uses LSTM layers with `EarlyStopping` to prevent overfitting. Initial training is conducted with a default learning rate.
5. **🔁 Reinforcement Study**: If the model's performance is unsatisfactory (i.e., MSE is above a certain threshold), it will:
    - Retrain with alternative learning rates (e.g., `[0.001, 0.005, 0.01]`).
    - Compare MSE scores across the different runs.
    - Choose the model configuration that yields the lowest MSE.
6. **📊 Signal Generation**: Compares the predicted price to the current price. If the change exceeds a small threshold, it generates a signal:
    - **Buy (1)** if the price is expected to rise.
    - **Sell (-1)** if the price is expected to fall.
    - **Hold (0)** if the change is neutral.

---

## 📌 Disclaimer

This toolkit is intended for **research and educational purposes only**. It is not a financial product and should not be used for actual trading without thorough personal testing and validation. Stock trading involves significant risk, and you should always consult with a qualified financial advisor before making any investment decisions.
