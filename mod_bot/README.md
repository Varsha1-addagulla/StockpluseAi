# McDonald's Stock Prediction Bot

This project predicts McDonald's (MCD) stock prices using a Machine Learning model (Random Forest) and simulates trading decisions based on those predictions.

## 📂 Project Structure

- **`main.py`**: The entry point. Fetches data, trains the model, runs the simulation, and plots results.
- **`data_loader.py`**: Handles downloading stock data from Yahoo Finance (`yfinance`) and preparing it for the model.
- **`model.py`**: Contains the `StockPredictor` class (Random Forest Regressor) that learns from historical data.
- **`trader.py`**: A simulation class that decides whether to BUY or SELL based on the model's predictions.
- **`requirements.txt`**: List of Python libraries required to run the bot.

## 🚀 How to Run

1.  **Open a terminal** in the project folder: `c:\Users\varsh\OneDrive\Desktop\mod_bot`
2.  **Activate the virtual environment**:
    ```powershell
    .\venv\Scripts\activate
    ```
3.  **Run the bot**:
    ```powershell
    python main.py
    ```
    - This will train the model on the latest data.
    - It will print the simulation results (Final Portfolio Value).
    - It will save a graph as `prediction_plot.png`.

## 🤖 How to Add Automation

### 1. Schedule Daily Runs (Windows Task Scheduler)
To make the bot run automatically every day (e.g., after market close):

1.  Open **Task Scheduler** on Windows.
2.  Click **Create Basic Task**.
3.  Name it "MCD Stock Bot".
4.  Trigger: **Daily**.
5.  Action: **Start a program**.
    - **Program/script**: `C:\Users\varsh\OneDrive\Desktop\mod_bot\venv\Scripts\python.exe`
    - **Add arguments**: `main.py`
    - **Start in**: `C:\Users\varsh\OneDrive\Desktop\mod_bot`

### 2. Real Trading (Next Steps)
Currently, `trader.py` only *simulates* trading (Paper Trading). To trade with real money:

1.  **Sign up for a Broker API**:
    - **Alpaca** (Great for algorithmic trading, offers a free paper trading API).
    - **Interactive Brokers**.
2.  **Update `trader.py`**:
    - Replace the `self.balance` logic with API calls to your broker to place real orders.
    - Example (Pseudo-code for Alpaca):
      ```python
      import alpaca_trade_api as tradeapi
      api = tradeapi.REST('KEY', 'SECRET', base_url='https://paper-api.alpaca.markets')
      api.submit_order(symbol='MCD', qty=1, side='buy', type='market', time_in_force='gtc')
      ```
