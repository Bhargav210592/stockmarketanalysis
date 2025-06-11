from flask import Flask, request, jsonify
from analyzer import StockAnalyzer, STOCK_CATEGORIES
from datetime import datetime

app = Flask(__name__)
analyzer = StockAnalyzer()

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()

        category = data.get("category")
        stock_type = data.get("stock_type", "Index")
        period = data.get("period", "60d")

        if category not in STOCK_CATEGORIES:
            return jsonify({"error": "Invalid category"}), 400

        if stock_type == "Index":
            ticker = STOCK_CATEGORIES[category]["ticker"]
            ticker_name = category
        else:
            stock_name = data.get("stock_name")
            stock_map = STOCK_CATEGORIES[category]["individual_stocks"]
            if stock_name not in stock_map:
                return jsonify({"error": "Invalid stock name"}), 400
            ticker = stock_map[stock_name]
            ticker_name = stock_name

        stock_data = analyzer.fetch_stock_data(ticker, period)
        if stock_data is None:
            return jsonify({"error": "Unable to fetch stock data"}), 500

        rsi = analyzer.compute_rsi(stock_data)
        macd = analyzer.compute_macd(stock_data)
        ma_data = analyzer.compute_moving_averages(stock_data)
        ma_data['current_price'] = stock_data['Close'].iloc[-1]

        headlines = analyzer.scrape_news_headlines(ticker_name)
        sentiment_label, sentiment_score = analyzer.analyze_sentiment(headlines)

        signal, confidence = analyzer.generate_signal(rsi, macd, ma_data, sentiment_score)
        ai_summary = analyzer.get_ai_summary(ticker_name, rsi, macd, ma_data, signal, confidence, sentiment_label, headlines)

        return jsonify({
            "ticker": ticker,
            "rsi": rsi,
            "macd": macd,
            "moving_averages": ma_data,
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment_score
            },
            "signal": {
                "recommendation": signal,
                "confidence": confidence
            },
            "ai_summary": ai_summary,
            "headlines": headlines[:5],
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
