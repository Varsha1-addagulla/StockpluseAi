from sklearn.linear_model import Ridge
import numpy as np
import joblib

class StockPredictor:
    def __init__(self, input_shape=None):
        # Ridge Regression: Fast, low memory, and effective for stock prediction
        # alpha=1.0: L2 regularization to prevent overfitting
        self.model = Ridge(alpha=1.0, random_state=42)

    def train(self, x_train, y_train, **kwargs):
        """Trains the model. x_train should be 2D: (samples, features*look_back)."""
        print(f"Training Ridge Regression on shape: {x_train.shape}")
        self.model.fit(x_train, y_train.ravel())

    def predict(self, x_test):
        """Makes predictions."""
        # Ensure 2D
        if len(x_test.shape) == 3:
            nsamples, nx, ny = x_test.shape
            x_test = x_test.reshape((nsamples, nx*ny))
            
        predictions = self.model.predict(x_test)
        return predictions.reshape(-1, 1)

    def predict_next_days(self, last_sequence, days=7, num_features=8):
        """
        Predicts the next 'days' stock prices recursively.
        last_sequence: The last known sequence of data (shape: (1, look_back * num_features))
        num_features: Number of features per time step (default 8 based on data_loader)
        """
        predictions = []
        # Ensure it's a flat vector (1 sample)
        current_sequence = last_sequence.flatten() 
        
        # Calculate look_back based on sequence length
        look_back = len(current_sequence) // num_features

        for _ in range(days):
            # Predict next step (Close price)
            # Model expects 2D array (1, n_features)
            input_seq = current_sequence.reshape(1, -1)
            next_pred_scaled = self.model.predict(input_seq)[0]
            predictions.append(next_pred_scaled)
            
            # Update sequence for next step
            # 1. Get the features of the very last time step in the sequence
            last_step_features = current_sequence[-num_features:].copy()
            
            # 2. Update the 'Close' price (index 0) with the prediction
            # We assume other technicals (RSI, etc.) stay similar to last day for short-term forecast
            # This is a simplification; ideally we'd predict them too or recalculate them
            last_step_features[0] = next_pred_scaled
            
            # 3. Shift window: Remove first time step (first 'num_features' elements)
            # and append the new step
            current_sequence = np.concatenate((current_sequence[num_features:], last_step_features))

        return np.array(predictions).reshape(-1, 1)
