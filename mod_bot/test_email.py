"""
Test script to verify email sending functionality
"""
from notifier import Notifier
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔧 Testing Email Integration...")
print("=" * 60)

# Initialize notifier
notifier = Notifier()

# Test email details
test_email = "varsha.addagulla@gmail.com"

# Test stock drop notification
stock_symbol = "MCD"
current_price = 285.50
predicted_price = 275.00

print(f"\n📉 Sending test stock drop notification for {stock_symbol}")
print(f"Current: ${current_price} -> Predicted: ${predicted_price}")

try:
    notifier.notify_decrease(
        user_email=test_email,
        stock=stock_symbol,
        price=current_price,
        predicted_price=predicted_price
    )
    print("\n✅ SUCCESS! Stock drop warning email sent!")
    print(f"📬 Check your inbox at {test_email}")
except Exception as e:
    print(f"\n❌ ERROR: Failed to send email")
    print(f"Error details: {e}")

print("\n" + "=" * 60)
