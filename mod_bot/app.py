from flask import Flask, render_template, url_for, flash, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
from flask_bcrypt import Bcrypt
from data_loader import DataLoader
from model import StockPredictor
from notifier import Notifier
import numpy as np
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import secrets

# Global Cache (AGGRESSIVELY optimized for low memory)
CACHE = {}
CACHE_EXPIRY = 45  # 45 seconds (further reduced)
MAX_CACHE_SIZE = 5  # Reduced to 5 entries max
MAX_DATE_RANGE_DAYS = 365  # 1 year maximum (reduced from 2)
BUFFER_DAYS = 50  # Reduced from 60

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Change this in production
# Database Configuration
database_url = os.environ.get('DATABASE_URL', 'sqlite:///site.db')
# Fix for Render's postgres:// usage (SQLAlchemy requires postgresql://)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Database Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)
    notify_sms = db.Column(db.Boolean, default=False)
    notify_email = db.Column(db.Boolean, default=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    avg_price = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Portfolio('{self.ticker}', qty={self.quantity})"

class PasswordResetToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    def is_expired(self):
        return datetime.utcnow() > self.expires_at
    
    def __repr__(self):
        return f"PasswordResetToken(user_id={self.user_id}, expires={self.expires_at})"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Global Objects ---
# In a real app, we'd load a pre-trained model. 
# For this demo, we'll train on startup if not exists (or just train quickly).
# Global variables for lazy initialization
predictor = None
scaler = None
notifier = Notifier()

def get_predictor():
    """Lazy initialization of the predictor model"""
    global predictor, scaler
    if predictor is None:
        print("Initializing Model...")
        loader = DataLoader(ticker='MCD', start_date='2015-01-01', end_date='2025-01-01')
        try:
            loader.fetch_data()
            x_train, y_train, scaler = loader.preprocess_data(look_back=60)
            predictor = StockPredictor(input_shape=(x_train.shape[1], 1))
            predictor.train(x_train, y_train)
            print("Model trained.")
        except Exception as e:
            print(f"Error initializing model: {e}")
            predictor = None
            scaler = None
    return predictor, scaler

# --- Routes ---
@app.route("/")
@app.route("/home")
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        phone = request.form.get('phone') # Get phone number
        password = request.form.get('password')
        
        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already taken. Please choose a different username.', 'danger')
            return render_template('register.html')
        
        # Check if email already exists
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered. Please use a different email or login.', 'danger')
            return render_template('register.html')
        
        # Check if phone number already exists
        if phone:  # Only check if phone number was provided
            existing_phone = User.query.filter_by(phone_number=phone).first()
            if existing_phone:
                flash('Phone number already registered. Please use a different phone number.', 'danger')
                return render_template('register.html')
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, phone_number=phone, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        
        # Send welcome email (skip if no email credentials)
        try:
            notifier.send_welcome_email(email, username)
        except Exception as e:
            print(f"Failed to send welcome email: {e}")
        
        flash('Your account has been created! You can now log in', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email_or_username = request.form.get('email') or request.form.get('username')
        password = request.form.get('password')
        # Try to find user by email first, then by username
        user = User.query.filter_by(email=email_or_username).first()
        if not user:
            user = User.query.filter_by(username=email_or_username).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check credentials', 'danger')
            
    return render_template('login.html')

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/forgot-password", methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate secure token
            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(minutes=15)
            
            # Delete any existing tokens for this user
            PasswordResetToken.query.filter_by(user_id=user.id).delete()
            
            # Create new reset token
            reset_token = PasswordResetToken(user_id=user.id, token=token, expires_at=expires_at)
            db.session.add(reset_token)
            db.session.commit()
            
            # Generate reset link
            reset_link = url_for('reset_password', token=token, _external=True)
            
            # Send password reset email
            notifier.send_password_reset(user.email, reset_link, user.username)
            
            flash('Password reset instructions have been sent to your email', 'info')
        else:
            # Don't reveal if email exists or not (security best practice)
            flash('If that email exists, password reset instructions have been sent', 'info')
        
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route("/reset-password/<token>", methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    # Find the reset token
    reset_token = PasswordResetToken.query.filter_by(token=token).first()
    
    if not reset_token or reset_token.is_expired():
        flash('Invalid or expired reset link', 'danger')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('reset_password.html', token=token)
        
        # Update user password
        user = User.query.get(reset_token.user_id)
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user.password = hashed_password
        
        # Delete the used token
        db.session.delete(reset_token)
        db.session.commit()
        
        flash('Your password has been reset! You can now log in', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

@app.route("/add_portfolio", methods=['POST'])
@login_required
def add_portfolio():
    ticker = request.form.get('ticker').upper()
    quantity = int(request.form.get('quantity'))
    price = float(request.form.get('price'))
    
    # Check if already exists, update if so
    existing = Portfolio.query.filter_by(user_id=current_user.id, ticker=ticker).first()
    if existing:
        total_cost = (existing.quantity * existing.avg_price) + (quantity * price)
        existing.quantity += quantity
        existing.avg_price = total_cost / existing.quantity
    else:
        new_holding = Portfolio(ticker=ticker, quantity=quantity, avg_price=price, user_id=current_user.id)
        db.session.add(new_holding)
    
    db.session.commit()
    flash(f'Added {quantity} shares of {ticker} to portfolio!', 'success')
    return redirect(url_for('portfolio'))

@app.route("/delete_portfolio/<int:id>")
@login_required
def delete_portfolio(id):
    holding = Portfolio.query.get_or_404(id)
    if holding.user_id == current_user.id:
        db.session.delete(holding)
        db.session.commit()
        flash('Stock removed from portfolio.', 'info')
    return redirect(url_for('portfolio'))

@app.route("/portfolio")
@login_required
def portfolio():
    portfolio_items = Portfolio.query.filter_by(user_id=current_user.id).all()
    # Calculate simple gain/loss for display (using stored avg_price vs last known price if available, else 0 change)
    # In a real app, we'd fetch real-time prices here. For now, we'll just pass the items.
    return render_template('portfolio.html', portfolio=portfolio_items)

@app.route("/settings", methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        current_user.phone_number = request.form.get('phone_number')
        current_user.notify_sms = 'notify_sms' in request.form
        current_user.notify_email = 'notify_email' in request.form
        db.session.commit()
        flash('Settings updated successfully!', 'success')
        return redirect(url_for('settings'))
    return render_template('settings.html')

@app.route("/dashboard", methods=['GET', 'POST'])
@login_required
def dashboard():
    # Default values
    ticker = 'MCD'
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d') # 2 years default
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    if request.method == 'POST':
        ticker = request.form.get('ticker').upper()
        form_start = request.form.get('start_date')
        form_end = request.form.get('end_date')
        quick_range = request.form.get('quick_range')  # New: quick select option
        
        # Handle quick range selection
        if quick_range:
            end_date = datetime.now().strftime('%Y-%m-%d')
            if quick_range == '1m':
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif quick_range == '3m':
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            elif quick_range == '6m':
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            elif quick_range == '1y':
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            elif quick_range == '2y':
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            elif quick_range == '5y':
                start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d')
        # Smart auto-fill: if only one date is provided
        elif form_start and not form_end:
            # User provided start date only → use today as end date
            start_date = form_start
            end_date = datetime.now().strftime('%Y-%m-%d')
        elif form_end and not form_start:
            # User provided end date only → calculate start date (2 years before)
            end_date = form_end
            end_dt = datetime.strptime(form_end, '%Y-%m-%d')
            start_date = (end_dt - timedelta(days=730)).strftime('%Y-%m-%d')
        elif form_start and form_end:
            # Both dates provided → validate
            start_dt = datetime.strptime(form_start, '%Y-%m-%d')
            end_dt = datetime.strptime(form_end, '%Y-%m-%d')
            days_diff = (end_dt - start_dt).days
            
            # Minimum 30 days (1 month) required
            if days_diff < 30:
                flash('⚠️ Date range too short. Minimum 30 days required. Using 1 month range.', 'warning')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
            # Warning if less than 2 years (but still allow it)
            elif days_diff < 730:
                flash('💡 For best predictions, we recommend at least 2 years of data. Short-term analysis may be less accurate.', 'info')
                start_date = form_start
                end_date = form_end
            else:
                # Good range (2+ years)
                start_date = form_start
                end_date = form_end
        else:
            # No dates provided, use defaults
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
    elif request.args.get('ticker'): # Allow GET request for "Analyze" button from portfolio
        ticker = request.args.get('ticker').upper()

    # Check Cache
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cache_key in CACHE:
        timestamp, cached_data = CACHE[cache_key]
        if time.time() - timestamp < CACHE_EXPIRY:
            print(f"Serving {ticker} from cache.")
            return render_template('dashboard.html', **cached_data)

    # Validate date range (prevent memory issues)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    days_range = (end_dt - start_dt).days
    
    if days_range > MAX_DATE_RANGE_DAYS:
        flash(f'⚠️ Maximum allowed range is {MAX_DATE_RANGE_DAYS//365} years. Using last 2 years.', 'warning')
        start_dt = end_dt - timedelta(days=MAX_DATE_RANGE_DAYS)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    # Initialize loader with BUFFERED dates for calculation
    # Using global BUFFER_DAYS constant (50 days)
    fetch_start_date = (start_dt - timedelta(days=BUFFER_DAYS)).strftime('%Y-%m-%d')
    
    current_loader = DataLoader(ticker=ticker, start_date=fetch_start_date, end_date=end_date)
    
    try:
        current_loader.fetch_data()
        if current_loader.data.empty:
            flash("No data found. Check ticker or date range.", "warning")
            return render_template('dashboard.html', price=0, prediction=0, trend="Error", 
                                   start_date=start_date, end_date=end_date, ticker=ticker, plot_url=None)
            
        # Get Sentiment
        try:
            sentiment_score, sentiment_label = current_loader.get_news_sentiment()
        except (ValueError, TypeError, IndexError):
            # Fallback if sentiment analysis fails
            sentiment_score, sentiment_label = 0.0, "Neutral"

        x_train, y_train, current_scaler = current_loader.preprocess_data(look_back=60)
        
        # Load pre-trained model from disk (MUCH faster than training)
        current_predictor = StockPredictor(input_shape=(x_train.shape[1], 1))
        model_path = os.path.join('models', 'predictor.joblib')
        
        # Try to load pre-trained model, fallback to training if not found
        if not current_predictor.load(model_path):
            print(f"Pre-trained model not found. Training new model for {ticker}...")
            current_predictor.train(x_train, y_train)

        
        # Get latest data point for prediction
        feature_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'Volatility', 'Market_Return', 'Volume_Change']
        last_60_features = current_loader.data[feature_cols].values[-60:]
        current_price = float(last_60_features[-1, 0]) # Close is at index 0
        
        # Scale the data (scaler expects 8 features)
        last_60_scaled = current_scaler.transform(last_60_features)
        
        # Flatten for model (1 sample, 60*8 features)
        X_test = last_60_scaled.flatten().reshape(1, -1)
        
        prediction_scaled = current_predictor.predict(X_test)
        
        # Inverse transform: we need to construct a dummy row with 8 cols to inverse transform the target
        # We only care about the first column (Close)
        dummy_row = np.zeros((1, len(feature_cols)))
        dummy_row[0, 0] = prediction_scaled[0]  # Fixed: prediction_scaled is 1D array
        # To inverse transform correctly, we ideally need the scaler to work on just target, 
        # but MinMaxScaler(feature_range) scales each feature independently.
        # So we can manually inverse transform if we know min/max of 'Close', OR
        # we can use the scaler on the dummy row and extract index 0.
        # However, standard scaler.inverse_transform expects (n_samples, n_features).
        predicted_price_raw = current_scaler.inverse_transform(dummy_row)[0, 0]
        
        predicted_price = float(predicted_price_raw)
        
        # --- 7-Day Forecast ---
        future_days = 7
        # predict_next_days expects the flattened sequence
        future_predictions_scaled = current_predictor.predict_next_days(X_test, days=future_days, num_features=len(feature_cols))
        
        future_predictions = []
        for p in future_predictions_scaled:
            dummy = np.zeros((1, len(feature_cols)))
            # Handle both scalar and array returns
            dummy[0, 0] = p if np.isscalar(p) else p[0]
            val = current_scaler.inverse_transform(dummy)[0, 0]
            future_predictions.append(val)
        
        future_predictions = np.array(future_predictions)
        
        # Generate Future Dates
        last_date = datetime.strptime(end_date, '%Y-%m-%d')
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(future_days)]
        
        # Calculate Confidence Score (Enhanced)
        # Factors: Volatility (Historical), Forecast Variance, Sentiment Strength
        hist_volatility = np.std(last_60_features[:, 0]) / np.mean(last_60_features[:, 0]) # Close price volatility
        forecast_volatility = np.std(future_predictions) / np.mean(future_predictions) if np.mean(future_predictions) != 0 else 0
        
        base_confidence = 95
        penalty = (hist_volatility * 100 * 2) + (forecast_volatility * 100 * 5)
            
        confidence_score = max(10, min(99, int(base_confidence - penalty)))
        
        # Get Technical Indicators (Latest)
        latest_rsi = current_loader.data['RSI_14'].iloc[-1]
        latest_macd = current_loader.data['MACD_12_26_9'].iloc[-1]
        latest_sma20 = current_loader.data['SMA_20'].iloc[-1]

        # Determine Trend & Send Alerts
        # We only trigger alerts if:
        # 1. The price change is significant (> 1%)
        # 2. The AI Confidence Score is high (> 75%) - Prevents "fake" triggers
        trend = "HOLD"
        
        if predicted_price > current_price * 1.01:
            trend = "UP"
            if current_user.notify_email and confidence_score > 75:
                print(f"🚀 High confidence ({confidence_score}%) buy signal for {ticker}. Sending email...")
                notifier.notify_increase(current_user.email, ticker, current_price, predicted_price)
                
        elif predicted_price < current_price * 0.99:
            trend = "DOWN"
            if current_user.notify_email and confidence_score > 75:
                print(f"⚠️ High confidence ({confidence_score}%) sell signal for {ticker}. Sending email...")
                notifier.notify_decrease(current_user.email, ticker, current_price, predicted_price)

        # Generate Plot
        # Filter data to show ONLY the user-selected range
        mask = (current_loader.data.index >= start_date) & (current_loader.data.index <= end_date)
        display_data = current_loader.data.loc[mask]
        
        if display_data.empty:
             # Fallback if mask is empty (shouldn't happen with 90d buffer, but safety first)
             display_data = current_loader.data.iloc[-30:] 
        
        real_prices = display_data['Close'].values.flatten() # Ensure 1D
        dates = display_data.index.strftime('%Y-%m-%d').tolist()
        
        plt.figure(figsize=(10, 5))
        
        # Plot Historical
        plt.plot(range(len(real_prices)), real_prices, label='Historical', color='#00d2ff')
        
        # Plot Forecast
        # Start forecast line from last real point to connect them
        forecast_x = range(len(real_prices)-1, len(real_prices) + future_days)
        forecast_y = np.concatenate(([real_prices[-1]], future_predictions))
        
        plt.plot(forecast_x, forecast_y, label='AI Forecast (7 Days)', color='#7000ff', linestyle='--', marker='o', markersize=4)
        
        plt.title(f'{ticker} Price Forecast', color='white')
        plt.xlabel('Time', color='white')
        plt.ylabel('Price', color='white')
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # Style for dark mode
        ax = plt.gca()
        ax.set_facecolor('#1a1a2e') # Match card bg roughly
        plt.gcf().set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        # Save to BytesIO
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

    except IndexError as e:
        # Specific error for insufficient data
        print(f"IndexError in dashboard: {e}")
        flash(f'⚠️ Not enough data for {ticker}. Please select a date range of at least 2 years for accurate predictions.', 'warning')
        return render_template('dashboard.html', price=0, prediction=0, trend="Error",
                               start_date=start_date, end_date=end_date, ticker=ticker, plot_url=None,
                               username=current_user.username, sentiment_label="Neutral", sentiment_score=0,
                               rsi=0, macd=0, sma=0, confidence=0, forecast_dates=[], forecast_prices=[])
    
    except ValueError as e:
        # Specific error for invalid ticker or data issues
        print(f"ValueError in dashboard: {e}")
        if "ticker" in str(e).lower() or "symbol" in str(e).lower():
            flash(f'❌ Invalid stock ticker "{ticker}". Please check the ticker symbol and try again.', 'danger')
        else:
            flash(f'⚠️ Data processing error for {ticker}. Try a different date range or ticker.', 'warning')
        return render_template('dashboard.html', price=0, prediction=0, trend="Error",
                               start_date=start_date, end_date=end_date, ticker=ticker, plot_url=None,
                               username=current_user.username, sentiment_label="Neutral", sentiment_score=0,
                               rsi=0, macd=0, sma=0, confidence=0, forecast_dates=[], forecast_prices=[])
    
    except Exception as e:
        # Generic error with helpful message
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in dashboard: {e}")
        print(f"Full traceback:\n{error_trace}")
        
        # Provide user-friendly error message
        error_msg = str(e)
        if "tuple index out of range" in error_msg or "index" in error_msg.lower():
            flash(f'⚠️ Insufficient data for {ticker}. Please select a longer date range (at least 2 years).', 'warning')
        elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            flash(f'🌐 Network error. Please check your internet connection and try again.', 'danger')
        else:
            flash(f'❌ Unable to analyze {ticker}. Please try a different stock or date range.', 'danger')
        
        return render_template('dashboard.html', price=0, prediction=0, trend="Error",
                               start_date=start_date, end_date=end_date, ticker=ticker, plot_url=None,
                               username=current_user.username, sentiment_label="Neutral", sentiment_score=0,
                               rsi=0, macd=0, sma=0, confidence=0, forecast_dates=[], forecast_prices=[])

    # Prepare data for template and cache
    template_data = {
        'price': round(float(current_price), 2),
        'prediction': round(float(predicted_price), 2),
        'trend': trend,
        'username': current_user.username,
        'start_date': start_date,
        'end_date': end_date,
        'ticker': ticker,
        'plot_url': plot_url,
        'sentiment_label': sentiment_label,
        'sentiment_score': round(sentiment_score, 2),
        'rsi': round(latest_rsi, 2),
        'macd': round(latest_macd, 2),
        'sma': round(latest_sma20, 2),
        'confidence': confidence_score,
        'forecast_dates': future_dates,
        'forecast_prices': [round(float(p), 2) for p in future_predictions]
    }
    
    # Store in Cache (with size limit) - Skip caching for large date ranges
    days_in_range = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
    
    if days_in_range <= 180:  # Only cache ranges <= 6 months
        if len(CACHE) >= MAX_CACHE_SIZE:
            # Remove oldest entry
            oldest_key = min(CACHE.keys(), key=lambda k: CACHE[k][0])
            CACHE.pop(oldest_key, None)
        
        CACHE[cache_key] = (time.time(), template_data)
    
    # Explicit memory cleanup
    import gc
    del current_loader, last_60_features, future_predictions
    plt.close('all')  # Close all matplotlib figures
    gc.collect()  # Force garbage collection

    return render_template('dashboard.html', **template_data)

@app.route("/test-email")
def test_email():
    """Test endpoint to verify SendGrid email sending."""
    to_email = request.args.get('to')
    
    if not to_email:
        return {"error": "Missing 'to' parameter. Usage: /test-email?to=email@example.com"}, 400
    
    # Send test email
    subject = "🧪 StockPulse AI - Test Email"
    html_content = """
    <html>
    <body style="font-family: Arial, sans-serif; padding: 20px;">
        <h2 style="color: #7000ff;">✅ Email Test Successful!</h2>
        <p>If you're reading this, SendGrid email integration is working correctly.</p>
        <p><strong>StockPulse AI</strong> is ready to send notifications!</p>
    </body>
    </html>
    """
    
    success = notifier.send_email(to_email, subject, html_content)
    
    if success:
        return {
            "status": "success",
            "message": f"Test email sent to {to_email}",
            "check": "Please check your inbox (and spam folder)"
        }, 200
    else:
        return {
            "status": "error",
            "message": "Failed to send email. Check SENDGRID_API_KEY and FROM_EMAIL environment variables."
        }, 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Migration: Add new columns if they don't exist
        from sqlalchemy import text
        try:
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE user ADD COLUMN notify_sms BOOLEAN DEFAULT 0"))
                conn.execute(text("ALTER TABLE user ADD COLUMN notify_email BOOLEAN DEFAULT 0"))
                print("Added notification columns to User table.")
        except Exception as e:
            # Columns likely already exist
            pass
    app.run(debug=True)
