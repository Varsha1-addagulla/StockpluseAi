from data_loader import DataLoader
from model import StockPredictor
import numpy as np

def test_pipeline():
    print("1. Testing Data Loader...")
    loader = DataLoader(ticker='MCD', start_date='2023-01-01', end_date='2023-06-01')
    loader.fetch_data()
    print(f"Data Shape: {loader.data.shape}")
    print(f"Columns: {loader.data.columns}")
    
    if 'SP500_Close' not in loader.data.columns:
        print("ERROR: SP500_Close not found!")
        return
        
    if 'Volatility' not in loader.data.columns:
        print("ERROR: Volatility not found!")
        return

    print("\n2. Testing Preprocessing...")
    x_train, y_train, scaler = loader.preprocess_data(look_back=60)
    print(f"X_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Check if 2D
    if len(x_train.shape) != 2:
        print(f"ERROR: x_train should be 2D, got {x_train.shape}")
        return

    print("\n3. Testing Model Training...")
    predictor = StockPredictor()
    predictor.train(x_train, y_train)
    print("Model trained successfully.")

    print("\n4. Testing Prediction...")
    # Create a dummy test sample (1 sample)
    x_test = x_train[-1].reshape(1, -1)
    prediction = predictor.predict(x_test)
    print(f"Prediction: {prediction}")
    
    print("\n5. Testing Recursive Forecast...")
    # Pass the last sequence from x_train (which is flattened)
    last_sequence = x_train[-1].reshape(1, -1)
    future_pred = predictor.predict_next_days(last_sequence, days=7, num_features=8)
    print(f"Future Predictions (7 days): {future_pred.flatten()}")
    
    print("\nSUCCESS: Pipeline verified!")

if __name__ == "__main__":
    test_pipeline()
