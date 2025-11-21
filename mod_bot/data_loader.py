import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob

class DataLoader:
    def __init__(self, ticker='MCD', start_date='2015-01-01', end_date='2025-01-01'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None

    def fetch_data(self):
        """Fetches historical data from Yahoo Finance, including S&P 500."""
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        
        # Flatten MultiIndex columns if present (yfinance update)
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)

        # Fetch S&P 500 for market context
        try:
            print("Fetching S&P 500 data...")
            sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date, progress=False)
            
            if not sp500.empty:
                # Handle MultiIndex for S&P 500 too
                if isinstance(sp500.columns, pd.MultiIndex):
                    sp500.columns = sp500.columns.get_level_values(0)
                
                if 'Close' in sp500.columns:
                    self.data = self.data.join(sp500['Close'].rename('SP500_Close'), how='left')
                    self.data['SP500_Close'] = self.data['SP500_Close'].ffill()
                else:
                    print("Warning: S&P 500 'Close' not found. Using flat market return.")
                    self.data['SP500_Close'] = self.data['Close']
            else:
                print("Warning: S&P 500 data empty. Using flat market return.")
                self.data['SP500_Close'] = self.data['Close'] # Fallback
        except Exception as e:
            print(f"Warning: Failed to fetch S&P 500 data: {e}")
            self.data['SP500_Close'] = self.data['Close'] # Fallback

        if self.data.empty:
            raise ValueError("No data fetched. Check ticker or date range.")
            
        # Calculate Technical Indicators
        self.calculate_indicators()
        
        print(f"Data fetched successfully. Shape: {self.data.shape}")
        return self.data

    def calculate_indicators(self):
        """Calculates RSI, MACD, SMA, Volatility, and Market Returns."""
        # SMA
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD_12_26_9'] = exp1 - exp2
        self.data['MACD_SIGNAL'] = self.data['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        
        # Volatility (20-day rolling std dev)
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std()
        
        # Market Return (S&P 500 daily change)
        self.data['Market_Return'] = self.data['SP500_Close'].pct_change()
        
        # Volume Change
        self.data['Volume_Change'] = self.data['Volume'].pct_change()
        
        # Fill NaNs
        self.data.bfill(inplace=True)
        self.data.ffill(inplace=True)

    def get_news_sentiment(self):
        """Fetches news and calculates average sentiment."""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            news = ticker_obj.news
            if not news:
                return 0, "Neutral"
            
            sentiments = []
            for article in news:
                title = article.get('title', '')
                blob = TextBlob(title)
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            if avg_sentiment > 0.1:
                label = "Positive"
            elif avg_sentiment < -0.1:
                label = "Negative"
            else:
                label = "Neutral"
                
            return avg_sentiment, label
        except Exception as e:
            print(f"Error fetching news: {e}")
            return 0, "Neutral"

    def preprocess_data(self, look_back=60):
        """
        Preprocesses data for Multivariate Gradient Boosting.
        Returns X (features) and y (target).
        """
        if self.data is None:
            self.fetch_data()

        # Select Features
        feature_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'Volatility', 'Market_Return', 'Volume_Change']
        data_features = self.data[feature_cols].values
        
        # Normalize the dataset
        scaled_data = self.scaler.fit_transform(data_features)
        
        x_train, y_train = [], []
        
        # Create sequences
        # For GB, we flatten the sequence: input is (look_back * num_features)
        for i in range(look_back, len(scaled_data)):
            x_train.append(scaled_data[i-look_back:i].flatten()) # Flatten time steps and features
            y_train.append(scaled_data[i, 0]) # Target is still 'Close' (index 0)
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        return x_train, y_train, self.scaler

if __name__ == "__main__":
    loader = DataLoader()
    data = loader.fetch_data()
    print(data.head())
