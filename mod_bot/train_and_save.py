#!/usr/bin/env python3
"""
Pre-train and save the stock prediction model to disk.
This script should be run once during deployment to generate model files.
"""

from data_loader import DataLoader
from model import StockPredictor
import joblib
import os

def train_and_save_model():
    """Train the model on default stock data and save to disk."""
    
    print("="*60)
    print("StockPulse AI - Model Pre-Training Script")
    print("="*60)
    
    # Configuration
    ticker = 'MCD'  # Default stock for training
    start_date = '2015-01-01'
    end_date = '2025-01-01'
    look_back = 60
    
    models_dir = 'models'
    predictor_path = os.path.join(models_dir, 'predictor.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    
    print(f"\n1. Loading data for {ticker} ({start_date} to {end_date})...")
    loader = DataLoader(ticker=ticker, start_date=start_date, end_date=end_date)
    
    try:
        loader.fetch_data()
        print(f"   ✓ Data fetched successfully. Shape: {loader.data.shape}")
        
        print(f"\n2. Preprocessing data (look_back={look_back})...")
        x_train, y_train, scaler = loader.preprocess_data(look_back=look_back)
        print(f"   ✓ Training data shape: {x_train.shape}")
        
        print(f"\n3. Training Ridge Regression model...")
        predictor = StockPredictor(input_shape=(x_train.shape[1], 1))
        predictor.train(x_train, y_train)
        print(f"   ✓ Model trained successfully")
        
        print(f"\n4. Saving model files...")
        predictor.save(predictor_path)
        joblib.dump(scaler, scaler_path)
        print(f"   ✓ Scaler saved to {scaler_path}")
        
        print("\n" + "="*60)
        print("SUCCESS! Pre-trained models saved:")
        print(f"  - {predictor_path}")
        print(f"  - {scaler_path}")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = train_and_save_model()
    exit(0 if success else 1)
