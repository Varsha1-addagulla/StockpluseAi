"""
Email notification service using SendGrid API.
Sends branded HTML emails for password resets, welcome messages, and stock alerts.
"""

import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from dotenv import load_dotenv

load_dotenv()

class Notifier:
    def __init__(self):
        """Initialize SendGrid notifier with API key and sender email."""
        self.api_key = os.getenv('SENDGRID_API_KEY')
        self.from_email = os.getenv('FROM_EMAIL', 'stockpulse56@gmail.com')
        
    def send_email(self, to_email, subject, html_content, plain_content=None):
        """
        Send email using SendGrid API.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email body
            plain_content: Plain text fallback (optional)
        """
        if not self.api_key:
            print("Warning: SENDGRID_API_KEY not set. Skipping email.")
            return False
            
        try:
            message = Mail(
                from_email=Email(self.from_email, 'StockPulse AI'),
                to_emails=To(to_email),
                subject=subject,
                html_content=Content("text/html", html_content)
            )
            
            if plain_content:
                message.add_content(Content("text/plain", plain_content))
            
            sg = SendGridAPIClient(self.api_key)
            response = sg.send(message)
            
            print(f"✓ Email sent to {to_email} (Status: {response.status_code})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to send email to {to_email}: {e}")
            return False
    
    def send_password_reset(self, user_email, reset_link, username="User"):
        """Send password reset email with branded HTML template."""
        
        subject = "🔐 Reset Your StockPulse AI Password"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #7000ff 0%, #00d2ff 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; background: #7000ff; color: white; 
                          padding: 15px 30px; text-decoration: none; border-radius: 5px; 
                          margin: 20px 0; font-weight: bold; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔐 Password Reset Request</h1>
                </div>
                <div class="content">
                    <p>Hi <strong>{username}</strong>,</p>
                    <p>We received a request to reset your password for your StockPulse AI account.</p>
                    <p>Click the button below to create a new password:</p>
                    <p style="text-align: center;">
                        <a href="{reset_link}" class="button">Reset My Password</a>
                    </p>
                    <p><strong>This link expires in 15 minutes.</strong></p>
                    <p>If you didn't request this password reset, you can safely ignore this email.</p>
                    <p>Best regards,<br><strong>The StockPulse AI Team</strong></p>
                </div>
                <div class="footer">
                    <p>© 2025 StockPulse AI</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        plain_content = f"""
        Password Reset Request
        
        Hi {username},
        
        We received a request to reset your password.
        Click this link: {reset_link}
        
        This link expires in 15 minutes.
        
        Best regards,
        The StockPulse AI Team
        """
        
        return self.send_email(user_email, subject, html_content, plain_content)

    def send_welcome_email(self, user_email, username):
        """Send welcome email to new users."""
        
        subject = "🎉 Welcome to StockPulse AI!"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #7000ff 0%, #00d2ff 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .feature {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .button {{ display: inline-block; background: #7000ff; color: white; 
                          padding: 15px 30px; text-decoration: none; border-radius: 5px; 
                          margin: 20px 0; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Welcome to StockPulse AI!</h1>
                </div>
                <div class="content">
                    <p>Hi <strong>{username}</strong>,</p>
                    <p>Thank you for joining StockPulse AI!</p>
                    
                    <h2 style="color: #7000ff;">What You Can Do:</h2>
                    <div class="feature">📈 Analyze stocks with AI predictions</div>
                    <div class="feature">📊 View 7-day price forecasts</div>
                    <div class="feature">💼 Manage your portfolio</div>
                    <div class="feature">🔔 Get email alerts</div>
                    
                    <p style="text-align: center;">
                        <a href="https://stockpluseai.onrender.com/dashboard" class="button">
                            Start Analyzing Stocks
                        </a>
                    </p>
                    
                    <p>Happy investing!<br><strong>The StockPulse AI Team</strong></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(user_email, subject, html_content)

    def notify_increase(self, user_email, stock, price, predicted_price):
        """Send notification when stock price is predicted to increase."""
        
        change_pct = ((predicted_price - price) / price) * 100
        
        subject = f"🚀 {stock} Alert: Price Predicted to Rise {change_pct:.1f}%!"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #00d2ff 0%, #00ff88 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .price-box {{ background: white; padding: 20px; margin: 20px 0; 
                             border-radius: 5px; text-align: center; }}
                .current {{ font-size: 24px; color: #666; }}
                .predicted {{ font-size: 32px; color: #00d2ff; font-weight: bold; }}
                .change {{ color: #00ff88; font-size: 20px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 {stock} Price Alert</h1>
                </div>
                <div class="content">
                    <p>Good news! <strong>{stock}</strong> is predicted to increase.</p>
                    
                    <div class="price-box">
                        <div class="current">Current: ${price:.2f}</div>
                        <div class="predicted">Predicted: ${predicted_price:.2f}</div>
                        <div class="change">▲ +{change_pct:.1f}%</div>
                    </div>
                    
                    <p><strong>💡 Suggestion:</strong> Consider buying now.</p>
                    <p style="font-size: 12px; color: #666;">
                        <em>This is an AI prediction, not financial advice.</em>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        plain_content = f"""
        Stock Alert: {stock}
        
        {stock} is predicted to rise.
        Current: ${price:.2f}
        Predicted: ${predicted_price:.2f}
        Change: +{change_pct:.1f}%
        
        Consider buying now.
        """
        
        return self.send_email(user_email, subject, html_content, plain_content)

    def notify_decrease(self, user_email, stock, price, predicted_price):
        """Send notification when stock price is predicted to decrease."""
        
        change_pct = abs(((predicted_price - price) / price) * 100)
        
        subject = f"⚠️ {stock} Alert: Price Predicted to Fall {change_pct:.1f}%"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #ff6b6b 0%, #ff9966 100%); 
                           color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .price-box {{ background: white; padding: 20px; margin: 20px 0; 
                             border-radius: 5px; text-align: center; }}
                .current {{ font-size: 24px; color: #666; }}
                .predicted {{ font-size: 32px; color: #ff6b6b; font-weight: bold; }}
                .change {{ color: #ff6b6b; font-size: 20px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚠️ {stock} Price Alert</h1>
                </div>
                <div class="content">
                    <p>Heads up! <strong>{stock}</strong> is predicted to decrease.</p>
                    
                    <div class="price-box">
                        <div class="current">Current: ${price:.2f}</div>
                        <div class="predicted">Predicted: ${predicted_price:.2f}</div>
                        <div class="change">▼ -{change_pct:.1f}%</div>
                    </div>
                    
                    <p><strong>💡 Suggestion:</strong> Consider selling or holding off.</p>
                    <p style="font-size: 12px; color: #666;">
                        <em>This is an AI prediction, not financial advice.</em>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        plain_content = f"""
        Stock Alert: {stock}
        
        {stock} is predicted to fall.
        Current: ${price:.2f}
        Predicted: ${predicted_price:.2f}
        Change: -{change_pct:.1f}%
        
        Consider selling or holding off.
        """
        
        return self.send_email(user_email, subject, html_content, plain_content)
