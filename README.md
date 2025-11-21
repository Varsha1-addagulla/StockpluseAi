# StockPulse AI - Stock Prediction Bot

This project predicts stock prices for **any** publicly traded company using a Machine Learning model (Random Forest) and simulates trading decisions based on those predictions.

## 📂 Project Structure

- **`app.py`**: The Flask web application for the dashboard and user interface.
- **`main.py`**: Command-line entry point for training and simulation.
- **`data_loader.py`**: Handles downloading stock data from Yahoo Finance (`yfinance`) and preparing it for the model.
- **`model.py`**: Contains the `StockPredictor` class (Random Forest Regressor) that learns from historical data.
- **`trader.py`**: A simulation class that decides whether to BUY or SELL based on the model's predictions.
- **`notifier.py`**: Handles email notifications for price alerts and password resets.
- **`requirements.txt`**: List of Python libraries required to run the bot.

## 🚀 How to Run

1.  **Open a terminal** in the project folder.
2.  **Activate the virtual environment**:
    ```powershell
    .\venv\Scripts\activate
    ```
3.  **Run the Web Dashboard**:
    ```powershell
    python app.py
    ```
    - Open your browser to `http://127.0.0.1:5000`
    - Register an account and log in.
    - Enter *any* stock ticker (e.g., AAPL, TSLA, NVDA) to analyze it.

## 🤖 Features

- **Multi-Stock Support**: Analyze any stock available on Yahoo Finance.
- **Machine Learning**: Uses Random Forest Regression to predict future prices.
- **Sentiment Analysis**: Analyzes news sentiment to adjust predictions.
- **Interactive Dashboard**: View historical data, predictions, and technical indicators.
- **Portfolio Tracking**: Simulate a portfolio and track gains/losses.
- **Email Notifications**: Get alerts when significant price movements are predicted.

## 🛠️ Automation

### Schedule Daily Runs
You can use Windows Task Scheduler to run the analysis scripts automatically.

### Real Trading
Currently, the bot *simulates* trading. To trade with real money, you would need to integrate a broker API like Alpaca or Interactive Brokers in `trader.py`.
