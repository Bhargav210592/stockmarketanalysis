import yfinance as yf
import numpy as np
import requests
from transformers import pipeline
import openai
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

class StockAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        self.setup_sentiment_analyzer()

    def setup_sentiment_analyzer(self):
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
        except Exception:
            self.sentiment_analyzer = pipeline("sentiment-analysis")

    def fetch_stock_data(self, ticker, period="60d"):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist if not hist.empty else None
        except:
            return None

    def compute_rsi(self, data, window=14):
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

    def compute_macd(self, data):
        try:
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            return {
                'line': macd.iloc[-1] if not np.isnan(macd.iloc[-1]) else 0.0,
                'signal': signal.iloc[-1] if not np.isnan(signal.iloc[-1]) else 0.0,
                'histogram': histogram.iloc[-1] if not np.isnan(histogram.iloc[-1]) else 0.0
            }
        except:
            return {'line': 0.0, 'signal': 0.0, 'histogram': 0.0}

    def compute_moving_averages(self, data):
        try:
            ma_20 = data['Close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else np.nan
            ma_25 = data['Close'].rolling(window=25).mean().iloc[-1] if len(data) >= 25 else np.nan
            ma_50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else np.nan
            ma_200 = data['Close'].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else np.nan
            return {
                'MA_20': ma_20 if not np.isnan(ma_20) else data['Close'].iloc[-1],
                'MA_25': ma_25 if not np.isnan(ma_25) else data['Close'].iloc[-1],
                'MA_50': ma_50 if not np.isnan(ma_50) else data['Close'].iloc[-1],
                'MA_200': ma_200 if not np.isnan(ma_200) else data['Close'].iloc[-1]
            }
        except:
            current_price = data['Close'].iloc[-1]
            return {'MA_20': current_price, 'MA_25': current_price, 'MA_50': current_price, 'MA_200': current_price}

    def scrape_news_headlines(self, ticker_name):
        try:
            query = ticker_name.replace("^", "").replace(".NS", "").replace("NSE", "")
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&pageSize=5"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            return [a["title"] for a in articles if len(a["title"]) > 15][:5] or [f"No news available for {query}"]
        except:
            return [f"News unavailable for {ticker_name}"]

    def analyze_sentiment(self, headlines):
        if not headlines or all("unavailable" in h.lower() for h in headlines):
            return "Neutral", 0

        sentiments = []
        try:
            for headline in headlines:
                result = self.sentiment_analyzer(headline)
                if isinstance(result[0], list):
                    scores = {r['label']: r['score'] for r in result[0]}
                    sentiments.append(scores.get('positive', 0) - scores.get('negative', 0))
                else:
                    label = result[0]['label'].upper()
                    score = result[0]['score']
                    sentiments.append(score if label == 'POSITIVE' else -score if label == 'NEGATIVE' else 0)

            avg_sentiment = np.mean(sentiments) if sentiments else 0
            if avg_sentiment > 0.1:
                return "Positive", avg_sentiment
            elif avg_sentiment < -0.1:
                return "Negative", avg_sentiment
            else:
                return "Neutral", avg_sentiment
        except:
            return "Neutral", 0

    def generate_signal(self, rsi, macd, ma_data, sentiment_score):
        signal = "Hold"
        confidence = "Low"
        bullish = bearish = 0

        if rsi < 40: bullish += 1
        elif rsi > 60: bearish += 1

        if macd['line'] > macd['signal'] and macd['histogram'] > 0: bullish += 1
        elif macd['line'] < macd['signal'] and macd['histogram'] < 0: bearish += 1

        price = ma_data['current_price']
        if price > ma_data['MA_25'] > ma_data['MA_50'] > ma_data['MA_200']: bullish += 1
        elif price < ma_data['MA_25'] < ma_data['MA_50'] < ma_data['MA_200']: bearish += 1

        if sentiment_score > 0.2: bullish += 0.5
        elif sentiment_score < -0.2: bearish += 0.5

        if bullish >= 2:
            signal = "Buy"
            confidence = "High" if bullish >= 2.5 else "Medium"
        elif bearish >= 2:
            signal = "Sell"
            confidence = "High" if bearish >= 2.5 else "Medium"

        return signal, confidence

    def get_ai_summary(self, ticker_name, rsi, macd, ma_data, signal, confidence, sentiment_label, headlines):
        prompt = f"""
        Analyze {ticker_name}:
        - RSI: {rsi:.2f} (Oversold<40, Overbought>60)
        - MACD Line: {macd['line']:.2f}
        - Signal: {macd['signal']:.2f}
        - Histogram: {macd['histogram']:.2f}
        - MAs: 20({ma_data['MA_20']:.2f}), 25({ma_data['MA_25']:.2f}), 50({ma_data['MA_50']:.2f}), 200({ma_data['MA_200']:.2f})
        - Signal: {signal} ({confidence})
        - Sentiment: {sentiment_label}
        - News: {'; '.join(headlines[:3])}
        Provide a brief strategy summary in 2-3 sentences.
        """
        try:
            if not OPENROUTER_API_KEY:
                raise Exception("API key not configured")

            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except:
            return f"Signal: {signal} with {confidence} confidence. RSI={rsi:.1f}, MACD={macd['line']:.2f} vs Signal={macd['signal']:.2f}, Sentiment={sentiment_label}."
