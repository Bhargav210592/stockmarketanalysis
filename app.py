import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import gspread
from google.oauth2.service_account import Credentials
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = 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
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NEWSAPI_KEY = 6cb55f32ee1a45bdbd625b3199605e0a

# Comprehensive Indian Stock Index and Individual Stock Mapping
STOCK_CATEGORIES = {
    "NIFTY 50 Index": {
        "ticker": "^NSEI",
        "individual_stocks": {
            "Reliance Industries": "RELIANCE.NS",
            "Tata Consultancy Services": "TCS.NS",
            "HDFC Bank": "HDFCBANK.NS",
            "Infosys": "INFY.NS",
            "ICICI Bank": "ICICIBANK.NS",
            "Hindustan Unilever": "HINDUNILVR.NS",
            "State Bank of India": "SBIN.NS",
            "ITC": "ITC.NS",
            "Bharti Airtel": "BHARTIARTL.NS",
            "Kotak Mahindra Bank": "KOTAKBANK.NS",
            "Larsen & Toubro": "LT.NS",
            "Axis Bank": "AXISBANK.NS",
            "Maruti Suzuki": "MARUTI.NS",
            "Asian Paints": "ASIANPAINT.NS",
            "Nestle India": "NESTLEIND.NS",
            "HCL Technologies": "HCLTECH.NS",
            "Bajaj Finance": "BAJFINANCE.NS",
            "Wipro": "WIPRO.NS",
            "Ultratech Cement": "ULTRACEMCO.NS",
            "Titan Company": "TITAN.NS",
            "Tata Motors": "TATAMOTORS.NS",
            "Sun Pharmaceutical": "SUNPHARMA.NS",
            "NTPC": "NTPC.NS",
            "Power Grid Corporation": "POWERGRID.NS",
            "Bajaj Finserv": "BAJAJFINSV.NS",
            "Dr. Reddy's Laboratories": "DRREDDY.NS",
            "Tech Mahindra": "TECHM.NS",
            "Oil & Natural Gas Corp": "ONGC.NS",
            "Tata Steel": "TATASTEEL.NS",
            "IndusInd Bank": "INDUSINDBK.NS",
            "Mahindra & Mahindra": "M&M.NS",
            "Adani Enterprises": "ADANIENT.NS",
            "Coal India": "COALINDIA.NS",
            "JSW Steel": "JSWSTEEL.NS",
            "Cipla": "CIPLA.NS",
            "Grasim Industries": "GRASIM.NS",
            "Hero MotoCorp": "HEROMOTOCO.NS",
            "UPL": "UPL.NS",
            "Britannia Industries": "BRITANNIA.NS",
            "Eicher Motors": "EICHERMOT.NS",
            "HDFC Life Insurance": "HDFCLIFE.NS",
            "SBI Life Insurance": "SBILIFE.NS",
            "Divis Laboratories": "DIVISLAB.NS",
            "Hindalco Industries": "HINDALCO.NS",
            "Bajaj Auto": "BAJAJ-AUTO.NS",
            "Shree Cement": "SHREECEM.NS",
            "Apollo Hospitals": "APOLLOHOSP.NS",
            "HDFC Asset Management": "HDFCAMC.NS",
            "Adani Ports": "ADANIPORTS.NS",
            "BPCL": "BPCL.NS"
        }
    },
    "BANK NIFTY Index": {
        "ticker": "^NSEBANK",
        "individual_stocks": {
            "HDFC Bank": "HDFCBANK.NS",
            "ICICI Bank": "ICICIBANK.NS",
            "State Bank of India": "SBIN.NS",
            "Kotak Mahindra Bank": "KOTAKBANK.NS",
            "Axis Bank": "AXISBANK.NS",
            "IndusInd Bank": "INDUSINDBK.NS",
            "Punjab National Bank": "PNB.NS",
            "Bank of Baroda": "BANKBARODA.NS",
            "Federal Bank": "FEDERALBNK.NS",
            "IDFC First Bank": "IDFCFIRSTB.NS",
            "AU Small Finance Bank": "AUBANK.NS",
            "Bandhan Bank": "BANDHANBNK.NS"
        }
    },
    "NIFTY AUTO Index": {
        "ticker": "^CNXAUTO",
        "individual_stocks": {
            "Maruti Suzuki": "MARUTI.NS",
            "Tata Motors": "TATAMOTORS.NS",
            "Mahindra & Mahindra": "M&M.NS",
            "Hero MotoCorp": "HEROMOTOCO.NS",
            "Eicher Motors": "EICHERMOT.NS",
            "Bajaj Auto": "BAJAJ-AUTO.NS",
            "TVS Motor Company": "TVSMOTOR.NS",
            "Ashok Leyland": "ASHOKLEY.NS",
            "Force Motors": "FORCEMOT.NS",
            "MRF": "MRF.NS",
            "Balkrishna Industries": "BALKRISIND.NS",
            "Ceat": "CEATLTD.NS",
            "Apollo Tyres": "APOLLOTYRE.NS",
            "JK Tyre": "JKTYRE.NS",
            "Bharat Forge": "BHARATFORG.NS"
        }
    },
    "NIFTY PHARMA Index": {
        "ticker": "^CNXPHARMA",
        "individual_stocks": {
            "Sun Pharmaceutical": "SUNPHARMA.NS",
            "Dr. Reddy's Laboratories": "DRREDDY.NS",
            "Cipla": "CIPLA.NS",
            "Divis Laboratories": "DIVISLAB.NS",
            "Lupin": "LUPIN.NS",
            "Aurobindo Pharma": "AUROPHARMA.NS",
            "Biocon": "BIOCON.NS",
            "Cadila Healthcare": "ZYDUSLIFE.NS",
            "Alkem Laboratories": "ALKEM.NS",
            "Glenmark Pharmaceuticals": "GLENMARK.NS",
            "Torrent Pharmaceuticals": "TORNTPHARM.NS",
            "Ipca Laboratories": "IPCALAB.NS",
            "Abbott India": "ABBOTINDIA.NS",
            "Pfizer": "PFIZER.NS",
            "GSK Pharma": "GSK.NS"
        }
    },
    "NIFTY METAL Index": {
        "ticker": "^CNXMETAL",
        "individual_stocks": {
            "Tata Steel": "TATASTEEL.NS",
            "JSW Steel": "JSWSTEEL.NS",
            "Hindalco Industries": "HINDALCO.NS",
            "Vedanta": "VEDL.NS",
            "Coal India": "COALINDIA.NS",
            "Steel Authority of India": "SAIL.NS",
            "NMDC": "NMDC.NS",
            "Jindal Steel & Power": "JINDALSTEL.NS",
            "National Aluminium Company": "NATIONALUM.NS",
            "APL Apollo Tubes": "APLAPOLLO.NS",
            "Hindustan Zinc": "HINDZINC.NS",
            "Ratnamani Metals": "RATNAMANI.NS",
            "Welspun Corp": "WELCORP.NS",
            "MOIL": "MOIL.NS"
        }
    },
    "NIFTY IT Index": {
        "ticker": "^CNXIT",
        "individual_stocks": {
            "Tata Consultancy Services": "TCS.NS",
            "Infosys": "INFY.NS",
            "HCL Technologies": "HCLTECH.NS",
            "Wipro": "WIPRO.NS",
            "Tech Mahindra": "TECHM.NS",
            "LTI Mindtree": "LTIM.NS",
            "Mphasis": "MPHASIS.NS",
            "Persistent Systems": "PERSISTENT.NS",
            "Coforge": "COFORGE.NS",
            "L&T Technology Services": "LTTS.NS"
        }
    },
    "NIFTY FMCG Index": {
        "ticker": "^CNXFMCG",
        "individual_stocks": {
            "Hindustan Unilever": "HINDUNILVR.NS",
            "ITC": "ITC.NS",
            "Nestle India": "NESTLEIND.NS",
            "Britannia Industries": "BRITANNIA.NS",
            "Dabur India": "DABUR.NS",
            "Marico": "MARICO.NS",
            "Godrej Consumer Products": "GODREJCP.NS",
            "Colgate-Palmolive": "COLPAL.NS",
            "United Spirits": "UBL.NS",
            "Tata Consumer Products": "TATACONSUM.NS",
            "Emami": "EMAMILTD.NS",
            "P&G Hygiene": "PGHH.NS",
            "VBL": "VBL.NS"
        }
    }
}

class StockAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        self.setup_sentiment_analyzer()
        
    def setup_sentiment_analyzer(self):
        """Initialize sentiment analysis pipeline"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
        except Exception as e:
            st.warning(f"Using default sentiment analyzer due to: {e}")
            self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def fetch_stock_data(self, ticker, period="60d"):
        """Fetch stock data using yfinance with longer period for MA calculations"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                st.error(f"No data found for ticker: {ticker}")
                return None
            return hist
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def compute_rsi(self, data, window=14):
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def compute_macd(self, data):
        """Calculate MACD"""
        try:
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            return {
                'line': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0,
                'signal': signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0.0,
                'histogram': histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
            }
        except:
            return {'line': 0.0, 'signal': 0.0, 'histogram': 0.0}
    
    def compute_moving_averages(self, data):
        """Calculate Moving Averages including 25-day MA"""
        try:
            ma_20 = data['Close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else np.nan
            ma_25 = data['Close'].rolling(window=25).mean().iloc[-1] if len(data) >= 25 else np.nan
            ma_50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else np.nan
            ma_200 = data['Close'].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else np.nan
            
            return {
                'MA_20': ma_20 if not pd.isna(ma_20) else data['Close'].iloc[-1],
                'MA_25': ma_25 if not pd.isna(ma_25) else data['Close'].iloc[-1],
                'MA_50': ma_50 if not pd.isna(ma_50) else data['Close'].iloc[-1],
                'MA_200': ma_200 if not pd.isna(ma_200) else data['Close'].iloc[-1]
            }
        except:
            current_price = data['Close'].iloc[-1]
            return {
                'MA_20': current_price,
                'MA_25': current_price,
                'MA_50': current_price,
                'MA_200': current_price
            }
    
    def scrape_news_headlines(self, ticker_name, days=1): # Defaulting to 1 day for recent news
        """Scrape news headlines from NewsAPI.org"""
        try:
            # It's good practice to store API keys in environment variables or a config file
            # For this example, I'm using the one you provided.
            # Consider moving this to a more secure location like an environment variable.
            api_key = NEWSAPI_KEY
            
            # Use the full company name for the query if available, otherwise use the ticker
            # This might require a mapping from ticker to full company name
            # For now, we'll use the ticker_name directly, cleaned up.
            search_query = ticker_name.replace("^", "").replace(".NS", "").replace("NSE", "")
            
            # Construct the NewsAPI URL
            # sortBy=publishedAt will give the most recent articles first
            # You can adjust pageSize if you need more or fewer headlines
            url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=publishedAt&apiKey={api_key}&pageSize=5"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            
            news_data = response.json()
            headlines = []
            
            if news_data.get("status") == "ok" and news_data.get("articles"):
                for article in news_data["articles"]:
                    title = article.get("title")
                    if title and len(title) > 15: # Basic filter for relevance
                        headlines.append(title)
                    if len(headlines) >= 5: # Limit to 5 headlines
                        break
            
            return headlines if headlines else [f"Recent market news for {search_query} unavailable via NewsAPI"]
            
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not fetch news from NewsAPI: {e}")
            return [f"Market news for {ticker_name} unavailable via NewsAPI"]
        except Exception as e:
            st.warning(f"An unexpected error occurred while fetching news: {e}")
            return [f"Market news for {ticker_name} unavailable via NewsAPI"]

    
    def analyze_sentiment(self, headlines):
        """Analyze sentiment of headlines"""
        if not headlines or all("unavailable" in h for h in headlines):
            return "Neutral", 0
        
        try:
            sentiments = []
            for headline in headlines:
                if self.sentiment_analyzer:
                    result = self.sentiment_analyzer(headline)
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            # FinBERT format
                            sentiment_scores = {item['label']: item['score'] for item in result[0]}
                            if 'positive' in sentiment_scores:
                                sentiments.append(sentiment_scores['positive'] - sentiment_scores.get('negative', 0))
                            else:
                                sentiments.append(0)
                        else:
                            # Standard format
                            label = result[0]['label'].upper()
                            score = result[0]['score']
                            if label in ['POSITIVE', 'POS']:
                                sentiments.append(score)
                            elif label in ['NEGATIVE', 'NEG']:
                                sentiments.append(-score)
                            else:
                                sentiments.append(0)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            if avg_sentiment > 0.1:
                return "Positive", avg_sentiment
            elif avg_sentiment < -0.1:
                return "Negative", avg_sentiment
            else:
                return "Neutral", avg_sentiment
                
        except Exception as e:
            st.warning(f"Sentiment analysis error: {e}")
            return "Neutral", 0
    
    def generate_signal(self, rsi, macd, ma_data, sentiment_score):
        """Generate trading signal based on technical indicators with updated RSI levels (40:60)"""
        signal = "Hold"
        confidence = "Low"
        
        # Signal generation logic
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI signals with new levels (40:60 instead of 30:70)
        if rsi < 40:
            bullish_signals += 1
        elif rsi > 60:
            bearish_signals += 1
        
        # MACD signals
        if macd['line'] > macd['signal'] and macd['histogram'] > 0:
            bullish_signals += 1
        elif macd['line'] < macd['signal'] and macd['histogram'] < 0:
            bearish_signals += 1
        
        # Moving Average signals (using 25-day MA as well)
        current_price = ma_data.get('current_price', ma_data['MA_20'])
        if (current_price > ma_data['MA_25'] > ma_data['MA_50'] and 
            ma_data['MA_25'] > ma_data['MA_200']):
            bullish_signals += 1
        elif (current_price < ma_data['MA_25'] < ma_data['MA_50'] and 
              ma_data['MA_25'] < ma_data['MA_200']):
            bearish_signals += 1
        
        # Sentiment boost
        if sentiment_score > 0.2:
            bullish_signals += 0.5
        elif sentiment_score < -0.2:
            bearish_signals += 0.5
        
        # Determine signal and confidence
        if bullish_signals >= 2:
            signal = "Buy"
            confidence = "High" if bullish_signals >= 2.5 else "Medium"
        elif bearish_signals >= 2:
            signal = "Sell"
            confidence = "High" if bearish_signals >= 2.5 else "Medium"
        else:
            signal = "Hold"
            confidence = "Medium" if abs(bullish_signals - bearish_signals) > 0.5 else "Low"
        
        return signal, confidence
    
    def get_ai_summary(self, ticker_name, rsi, macd, ma_data, signal, confidence, sentiment_label, headlines):
        """Generate AI summary using OpenRouter DeepSeek"""
        try:
            # Prepare the prompt
            prompt = f"""
            You are a professional stock market analyst. Analyze the following data for {ticker_name}:

            Technical Indicators:
            - RSI: {rsi:.2f} (Oversold<40, Overbought>60)
            - MACD Line: {macd['line']:.2f}
            - MACD Signal: {macd['signal']:.2f}
            - MACD Histogram: {macd['histogram']:.2f}
            - 20-day MA: {ma_data['MA_20']:.2f}
            - 25-day MA: {ma_data['MA_25']:.2f}
            - 50-day MA: {ma_data['MA_50']:.2f}
            - 200-day MA: {ma_data['MA_200']:.2f}

            Generated Signal: {signal}
            Confidence: {confidence}
            News Sentiment: {sentiment_label}

            Recent Headlines (Last 3 days):
            {chr(10).join(headlines[:3])}

            Provide a concise trading strategy summary (2-3 sentences) explaining:
            1. Why this signal was generated
            2. Key risk factors to consider
            3. Suggested entry/exit strategy if applicable

            Keep it professional and actionable for retail traders.
            """
            
            # OpenRouter API call
            if OPENROUTER_API_KEY:
                client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                )
                
                response = client.chat.completions.create(
                    model="deepseek/deepseek-chat",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
            else:
                raise Exception("API key not configured")
            
        except Exception as e:
            st.warning(f"AI summary unavailable: {e}")
            rsi_status = "oversold" if rsi < 40 else "overbought" if rsi > 60 else "neutral"
            macd_trend = "bullish" if macd['line'] > macd['signal'] else "bearish"
            return f"Technical analysis suggests {signal} signal with {confidence} confidence. RSI at {rsi:.1f} indicates {rsi_status} conditions and MACD shows {macd_trend} momentum. Consider market sentiment and risk management."

def setup_google_sheets():
    """Setup Google Sheets connection"""
    try:
        # Define the scope
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Load credentials from service account file
        creds = Credentials.from_service_account_file(
            "service_account.json", 
            scopes=scope
        )
        
        client = gspread.authorize(creds)
        
        # Try to open existing sheet or create new one
        try:
            sheet = client.open("StockAnalyzerLog").sheet1
        except gspread.SpreadsheetNotFound:
            # Create new spreadsheet
            spreadsheet = client.create("StockAnalyzerLog")
            sheet = spreadsheet.sheet1
            
            # Add headers
            headers = [
                "Timestamp", "Ticker", "RSI", "MACD_Line", "MACD_Signal", 
                "MA_20", "MA_25", "MA_50", "MA_200", "Signal", "Confidence", 
                "AI_Summary", "News_Sentiment"
            ]
            sheet.append_row(headers)
        
        return sheet
    except Exception as e:
        st.error(f"Google Sheets setup failed: {e}")
        return None

def log_to_sheets(sheet, data):
    """Log analysis results to Google Sheets"""
    if sheet:
        try:
            sheet.append_row(data)
            return True
        except Exception as e:
            st.warning(f"Failed to log to sheets: {e}")
            return False
    return False

def send_email_alert(ticker_name, signal, confidence, ai_summary, sentiment_label):
    """Send email alert"""
    try:
        if not GMAIL_EMAIL or not GMAIL_APP_PASSWORD:
            st.warning("Email credentials not configured")
            return False
            
        msg = MIMEMultipart()
        msg['From'] = GMAIL_EMAIL
        msg['To'] = GMAIL_EMAIL  # Sending to self for demo
        msg['Subject'] = f"{confidence}-Confidence {signal.upper()} Alert: {ticker_name}"
        
        body = f"""
        Stock Analysis Alert
        
        Index/Stock: {ticker_name}
        Signal: {signal.upper()}
        Confidence: {confidence}
        
        AI Strategy Summary:
        {ai_summary}
        
        News Sentiment: {sentiment_label}
        
        Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        This is an automated alert from your Stock Analyzer AI Agent.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        st.warning(f"Email sending failed: {e}")
        return False

def create_indicator_charts(data, rsi, macd, ma_data):
    """Create indicator charts with improved MA display"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Price and Moving Averages
    axes[0].plot(data.index[-30:], data['Close'][-30:], label='Close Price', linewidth=2, color='black')
    
    # Calculate and plot moving averages for the chart
    if len(data) >= 20:
        ma_20_series = data['Close'].rolling(window=20).mean()
        axes[0].plot(data.index[-30:], ma_20_series[-30:], color='orange', linestyle='--', label='MA 20', alpha=0.8)
    
    if len(data) >= 25:
        ma_25_series = data['Close'].rolling(window=25).mean()
        axes[0].plot(data.index[-30:], ma_25_series[-30:], color='purple', linestyle='--', label='MA 25', alpha=0.8)
    
    if len(data) >= 50:
        ma_50_series = data['Close'].rolling(window=50).mean()
        axes[0].plot(data.index[-30:], ma_50_series[-30:], color='blue', linestyle='--', label='MA 50', alpha=0.8)
    
    if len(data) >= 200:
        ma_200_series = data['Close'].rolling(window=200).mean()
        axes[0].plot(data.index[-30:], ma_200_series[-30:], color='red', linestyle='--', label='MA 200', alpha=0.8)
    
    axes[0].set_title('Price & Moving Averages')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RSI with updated levels (40:60)
    rsi_series = []
    for i in range(len(data)):
        if i >= 13:  # Need at least 14 points for RSI
            temp_data = data.iloc[:i+1]
            delta = temp_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            temp_rsi = 100 - (100 / (1 + rs))
            rsi_series.append(temp_rsi.iloc[-1] if not pd.isna(temp_rsi.iloc[-1]) else 50)
        else:
            rsi_series.append(50)
    
    axes[1].plot(data.index[-30:], rsi_series[-30:], label='RSI', color='purple', linewidth=2)
    axes[1].axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Overbought (60)')
    axes[1].axhline(y=40, color='green', linestyle='--', alpha=0.7, label='Oversold (40)')
    axes[1].axhline(y=50, color='gray', linestyle='-', alpha=0.5, label='Neutral (50)')
    axes[1].set_title(f'RSI (Current: {rsi:.2f})')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MACD
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    
    axes[2].plot(data.index[-30:], macd_line[-30:], label='MACD Line', color='blue')
    axes[2].plot(data.index[-30:], signal_line[-30:], label='Signal Line', color='red')
    axes[2].bar(data.index[-30:], histogram[-30:], label='Histogram', alpha=0.3, color='gray')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_title('MACD')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(
        page_title="📈 Stock Market Analyzer AI",
        page_icon="📈",
        layout="wide"
    )
    
    # Title and description
    st.title("📈 Stock Market Analyzer AI Agent")
    st.markdown("*Intelligent analysis for Indian stock indices and individual stocks with AI-powered insights*")
    
    # Sidebar
    st.sidebar.header("Stock Selection")
    
    # Category selection
    selected_category = st.sidebar.selectbox(
        "Choose Category:",
        list(STOCK_CATEGORIES.keys()),
        index=0
    )
    
    # Stock type selection
    stock_type = st.sidebar.radio(
        "Analysis Type:",
        ["Index", "Individual Stock"]
    )
    
    if stock_type == "Index":
        ticker = STOCK_CATEGORIES[selected_category]["ticker"]
        ticker_name = selected_category
    else:
        # Individual stock selection
        individual_stocks = STOCK_CATEGORIES[selected_category]["individual_stocks"]
        selected_stock = st.sidebar.selectbox(
            "Choose Stock:",
            list(individual_stocks.keys())
        )
        ticker = individual_stocks[selected_stock]
        ticker_name = selected_stock
    
    # Display current selection
    st.sidebar.success(f"Selected: {ticker_name}")
    st.sidebar.info(f"Ticker: {ticker}")
    
    # Analysis settings
    st.sidebar.header("Settings")
    period = st.sidebar.selectbox(
        "Data Period:",
        ["30d", "60d", "90d", "6mo", "1y"],
        index=1
    )
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StockAnalyzer()
    
    # Setup Google Sheets
    #if 'sheets_client' not in st.session_state:
        #st.session_state.sheets_client = setup_google_sheets()
    
    # Main analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Now", type="primary", use_container_width=True)
    
    if analyze_button:
        with st.spinner(f"Analyzing {ticker_name}..."):
            try:
                # Fetch data
                data = st.session_state.analyzer.fetch_stock_data(ticker, period)
                if data is None:
                    st.stop()
                
                # Calculate indicators
                rsi = st.session_state.analyzer.compute_rsi(data)
                macd = st.session_state.analyzer.compute_macd(data)
                ma_data = st.session_state.analyzer.compute_moving_averages(data)
                
                # Add current price to ma_data for signal generation
                ma_data['current_price'] = data['Close'].iloc[-1]
                
                # Get news and sentiment (24 hours)
                headlines = st.session_state.analyzer.scrape_news_headlines(ticker_name, days=1)
                sentiment_label, sentiment_score = st.session_state.analyzer.analyze_sentiment(headlines)
                
                # Generate signal
                signal, confidence = st.session_state.analyzer.generate_signal(
                    rsi, macd, ma_data, sentiment_score
                )
                
                # Get AI summary
                ai_summary = st.session_state.analyzer.get_ai_summary(
                    ticker_name, rsi, macd, ma_data, signal, confidence, 
                    sentiment_label, headlines
                )
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Current price display
                current_price = data['Close'].iloc[-1]
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"₹{current_price:.2f}",
                        delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                    )
                
                with col2:
                    signal_color = {"Buy": "🟢", "Sell": "🔴", "Hold": "🟡"}[signal]
                    st.metric(
                        label="Trading Signal",
                        value=f"{signal_color} {signal}",
                        delta=f"Confidence: {confidence}"
                    )
                
                with col3:
                    rsi_status = "Oversold" if rsi < 40 else "Overbought" if rsi > 60 else "Normal"
                    st.metric(
                        label="RSI (40:60)",
                        value=f"{rsi:.2f}",
                        delta=rsi_status
                    )
                
                with col4:
                    sentiment_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}[sentiment_label]
                    st.metric(
                        label="News Sentiment (Recent)", # Or "News Sente# 24 hours)"
                       value=f"{sentiment_color} {sentiment_label}",
                        delta=f"Score: {sentiment_score:.2f}"
                    )
                
                # AI Strategy Summary
                st.subheader("🤖 AI Strategy Summary")
                st.info(ai_summary)
                
                # Technical Indicators
                st.subheader("📊 Technical Indicators")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Moving Averages:**")
                    st.write(f"• 20-day MA: ₹{ma_data['MA_20']:.2f}")
                    st.write(f"• 25-day MA: ₹{ma_data['MA_25']:.2f}")
                    st.write(f"• 50-day MA: ₹{ma_data['MA_50']:.2f}")
                    st.write(f"• 200-day MA: ₹{ma_data['MA_200']:.2f}")
                    
                    # MA trend analysis
                    if ma_data['MA_20'] > ma_data['MA_50'] > ma_data['MA_200']:
                        st.success("📈 Strong Bullish MA Alignment")
                    elif ma_data['MA_20'] < ma_data['MA_50'] < ma_data['MA_200']:
                        st.error("📉 Strong Bearish MA Alignment")
                    else:
                        st.warning("🔄 Mixed MA Signals")
                
                with col2:
                    st.write("**MACD:**")
                    st.write(f"• MACD Line: {macd['line']:.4f}")
                    st.write(f"• Signal Line: {macd['signal']:.4f}")
                    st.write(f"• Histogram: {macd['histogram']:.4f}")
                    
                    # MACD trend analysis
                    if macd['line'] > macd['signal'] and macd['histogram'] > 0:
                        st.success("📈 Bullish MACD Crossover")
                    elif macd['line'] < macd['signal'] and macd['histogram'] < 0:
                        st.error("📉 Bearish MACD Crossover")
                    else:
                        st.warning("🔄 MACD Consolidation")
                
                # Charts
                st.subheader("📈 Technical Analysis Charts")
                fig = create_indicator_charts(data, rsi, macd, ma_data)
                st.pyplot(fig)
                
                # Volume analysis
                st.subheader("📊 Volume Analysis")
                avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Current Volume",
                        value=f"{current_volume:,.0f}",
                        delta=f"{volume_ratio:.2f}x Average"
                    )
                
                with col2:
                    if volume_ratio > 1.5:
                        st.success("🔊 High Volume Activity")
                    elif volume_ratio < 0.5:
                        st.warning("🔇 Low Volume Activity")
                    else:
                        st.info("🔉 Normal Volume Activity")
                
                # News Headlines
                st.subheader("📰 Recent News Headlines (Last 3 Days)")
                for i, headline in enumerate(headlines[:5], 1):
                    if "unavailable" not in headline.lower():
                        st.write(f"{i}. {headline}")
                    else:
                        st.write(f"{i}. {headline}")
                
                # Support and Resistance Levels
                st.subheader("📏 Support & Resistance Levels")
                
                # Calculate support and resistance
                high_20 = data['High'].rolling(window=20).max().iloc[-1]
                low_20 = data['Low'].rolling(window=20).min().iloc[-1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("20-Day High", f"₹{high_20:.2f}")
                with col2:
                    st.metric("20-Day Low", f"₹{low_20:.2f}")
                with col3:
                    range_pct = ((high_20 - low_20) / low_20) * 100
                    st.metric("Range", f"{range_pct:.1f}%")
                
                # Log to Google Sheets
                log_data = [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    ticker_name,
                    f"{rsi:.2f}",
                    f"{macd['line']:.4f}",
                    f"{macd['signal']:.4f}",
                    f"{ma_data['MA_20']:.2f}",
                    f"{ma_data['MA_25']:.2f}",
                    f"{ma_data['MA_50']:.2f}",
                    f"{ma_data['MA_200']:.2f}",
                    signal,
                    confidence,
                    ai_summary[:500],  # Limit length
                    sentiment_label
                ]
                
                #if log_to_sheets(st.session_state.sheets_client, log_data):
                    #st.success("✅ Results logged to Google Sheets")
                
                # Email alert option
                if confidence in ["High", "Medium"]:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📧 Send Email Alert"):
                            if send_email_alert(ticker_name, signal, confidence, ai_summary, sentiment_label):
                                st.success("✅ Email alert sent successfully!")
                            else:
                                st.error("❌ Failed to send email alert")
                    
                    with col2:
                        # Download analysis report
                        report_data = {
                            "Stock": ticker_name,
                            "Analysis_Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "Current_Price": current_price,
                            "Signal": signal,
                            "Confidence": confidence,
                            "RSI": rsi,
                            "MACD": macd,
                            "Moving_Averages": ma_data,
                            "Sentiment": sentiment_label,
                            "AI_Summary": ai_summary
                        }
                        
                        if st.download_button(
                            label="📥 Download Report",
                            data=json.dumps(report_data, indent=2),
                            file_name=f"{ticker_name}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        ):
                            st.success("✅ Report downloaded!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.write("Please check your internet connection and API credentials.")
                st.exception(e)
    
    # Information panel
    with st.expander("ℹ️ About This Analyzer"):
        st.write("""
        **Enhanced Features:**
        - ✅ Updated RSI levels (40:60 instead of 30:70)
        - ✅ Added 25-day Moving Average
        - ✅ Fixed Moving Average calculations
        - ✅ 3-day news headlines
        - ✅ Complete Nifty stock coverage
        - ✅ Individual stock analysis
        - ✅ Enhanced volume analysis
        - ✅ Support/Resistance levels
        
        **Technical Indicators:**
        - **RSI (40:60)**: Oversold below 40, Overbought above 60
        - **MACD**: Momentum indicator with signal line crossovers
        - **Moving Averages**: 20, 25, 50, and 200-day periods
        - **Volume**: Compared to 20-day average
        
        **Signal Generation:**
        Combines technical indicators with news sentiment for comprehensive analysis.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "*Enhanced Stock Analyzer powered by yfinance, FinBERT, OpenRouter DeepSeek, and Google Sheets API*"
    )
    st.markdown("*Updated with RSI (40:60), 25-day MA, 3-day news, and complete Nifty stock coverage*")

if __name__ == "__main__":
    main()
