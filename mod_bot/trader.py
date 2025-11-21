import numpy as np
from notifier import Notifier

class Trader:
    def __init__(self, initial_balance=10000, user_email=None):
        self.balance = initial_balance
        self.holdings = 0
        self.portfolio_value = initial_balance
        self.history = []
        self.notifier = Notifier()
        self.user_email = user_email

    def decide(self, current_price, predicted_price):
        """
        Simple logic: 
        - Buy if predicted price > current price (and we have cash)
        - Sell if predicted price < current price (and we have holdings)
        """
        action = "HOLD"
        
        if predicted_price > current_price * 1.01: # 1% threshold
            if self.balance > current_price:
                self.buy(current_price)
                action = "BUY"
                if self.user_email:
                    self.notifier.notify_increase(self.user_email, "MCD", current_price, predicted_price)
        elif predicted_price < current_price * 0.99:
            if self.holdings > 0:
                self.sell(current_price)
                action = "SELL"
                if self.user_email:
                    self.notifier.notify_decrease(self.user_email, "MCD", current_price, predicted_price)
                
        return action

    def buy(self, price):
        # Buy as much as possible
        shares_to_buy = self.balance // price
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            self.balance -= cost
            self.holdings += shares_to_buy
            self.history.append({'action': 'BUY', 'price': price, 'shares': shares_to_buy})
            print(f"BUY: {shares_to_buy} shares at {price:.2f}")

    def sell(self, price):
        # Sell all
        if self.holdings > 0:
            revenue = self.holdings * price
            self.balance += revenue
            print(f"SELL: {self.holdings} shares at {price:.2f}")
            self.history.append({'action': 'SELL', 'price': price, 'shares': self.holdings})
            self.holdings = 0

    def get_portfolio_value(self, current_price):
        return self.balance + (self.holdings * current_price)
