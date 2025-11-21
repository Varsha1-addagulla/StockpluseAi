import numpy as np
import pandas as pd
from data_loader import DataLoader
from model import StockPredictor
from trader import Trader
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    # 1. Data Acquisition & Preprocessing
    loader = DataLoader(ticker='MCD', start_date='2015-01-01', end_date='2025-01-01')
    try:
        raw_data = loader.fetch_data()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    look_back = 60
    x_train, y_train, scaler = loader.preprocess_data(look_back=look_back)
    
    # 2. Model Development
    print("Building and training model...")
    predictor = StockPredictor(input_shape=(x_train.shape[1], 1))
    predictor.train(x_train, y_train) 
    
    # 3. Simulation / Evaluation (using training data for now as a demo)
    print("Running simulation...")
    predictions = predictor.predict(x_train)
    predicted_prices = scaler.inverse_transform(predictions)
    real_prices = scaler.inverse_transform(y_train.reshape(-1, 1))
    
    # Get email from env or hardcode for testing
    user_email = os.getenv('USER_EMAIL_FOR_NOTIFICATIONS')
    trader = Trader(user_email=user_email)
    
    # Simulate trading
    for i in range(len(predicted_prices)):
        current_price = real_prices[i][0]
        predicted_next = predicted_prices[i][0] 
        
        trader.decide(current_price, predicted_next)
        
    final_value = trader.get_portfolio_value(real_prices[-1][0])
    print(f"Final Portfolio Value: ${final_value:.2f}")

    # Plotting
    plt.figure(figsize=(14,5))
    plt.plot(real_prices, color='red', label='Real McDonald\'s Stock Price')
    plt.plot(predicted_prices, color='blue', label='Predicted McDonald\'s Stock Price')
    plt.title('McDonald\'s Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('prediction_plot.png')
    print("Plot saved to prediction_plot.png")

if __name__ == "__main__":
    main()
