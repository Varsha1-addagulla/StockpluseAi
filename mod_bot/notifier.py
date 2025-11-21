import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

class Notifier:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.sender_password = os.getenv('SENDER_PASSWORD')

    def send_email(self, recipient_email, subject, body):
        if not self.sender_email or not self.sender_password:
            print("Warning: Email credentials not set. Skipping email.")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, recipient_email, text)
            server.quit()
            print(f"Email sent to {recipient_email}")
        except Exception as e:
            print(f"Failed to send email: {e}")

    def notify_increase(self, user_email, stock, price, predicted_price):
        subject = f"🚀 {stock} Opportunity: Price Increasing!"
        body = f"Good news!\n\n{stock} is currently at ${price:.2f} and is predicted to rise to ${predicted_price:.2f}.\n\nConsider buying now."
        self.send_email(user_email, subject, body)

    def notify_decrease(self, user_email, stock, price, predicted_price):
        subject = f"⚠️ {stock} Warning: Price Falling!"
        body = f"Heads up!\n\n{stock} is currently at ${price:.2f} and is predicted to drop to ${predicted_price:.2f}.\n\nConsider selling or holding off."
        self.send_email(user_email, subject, body)

    def send_password_reset(self, user_email, reset_link, username="User"):
        """Send password reset email with HTML formatting"""
        if not self.sender_email or not self.sender_password:
            print(f"\n{'='*60}")
            print(f"PASSWORD RESET LINK FOR {user_email}")
            print(f"{'='*60}")
            print(f"{reset_link}")
            print(f"Token expires in 15 minutes")
            print(f"{'='*60}\n")
            return

        subject = "🔐 StockPulse AI - Password Reset Request"
        
        # HTML email body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f4; margin: 0; padding: 0; }}
                .container {{ max-width: 600px; margin: 40px auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%); padding: 30px; text-align: center; }}
                .header h1 {{ color: white; margin: 0; font-size: 24px; }}
                .content {{ padding: 40px 30px; }}
                .content p {{ color: #333; line-height: 1.6; margin: 15px 0; }}
                .button {{ display: inline-block; background: linear-gradient(135deg, #4A00E0 0%, #8E2DE2 100%); color: white; padding: 14px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; margin: 20px 0; }}
                .button:hover {{ opacity: 0.9; }}
                .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 12px; }}
                .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 20px 0; color: #856404; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔐 Password Reset Request</h1>
                </div>
                <div class="content">
                    <p>Hi {username},</p>
                    <p>We received a request to reset your password for your StockPulse AI account.</p>
                    <p>Click the button below to create a new password:</p>
                    <p style="text-align: center;">
                        <a href="{reset_link}" class="button">Reset My Password</a>
                    </p>
                    <div class="warning">
                        <strong>⏱️ This link expires in 15 minutes</strong>
                    </div>
                    <p>If you didn't request this password reset, you can safely ignore this email. Your password will remain unchanged.</p>
                    <p>For security reasons, this link can only be used once.</p>
                    <p style="margin-top: 30px;">Best regards,<br><strong>The StockPulse AI Team</strong></p>
                </div>
                <div class="footer">
                    <p>© 2025 StockPulse AI. Powered by Advanced Analytics.</p>
                    <p>This is an automated message, please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text fallback
        text_body = f"""
Hi {username},

We received a request to reset your password for your StockPulse AI account.

Click the link below to create a new password:
{reset_link}

⏱️ This link expires in 15 minutes.

If you didn't request this password reset, you can safely ignore this email.

Best regards,
The StockPulse AI Team
        """

        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = user_email
            msg['Subject'] = subject
            
            # Attach both plain text and HTML versions
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, user_email, msg.as_string())
            server.quit()
            print(f"Password reset email sent to {user_email}")
        except Exception as e:
            print(f"Failed to send password reset email: {e}")
            # Fallback to console output
            print(f"\n{'='*60}")
            print(f"PASSWORD RESET LINK FOR {user_email}")
            print(f"{'='*60}")
            print(f"{reset_link}")
            print(f"Token expires in 15 minutes")
            print(f"{'='*60}\n")
