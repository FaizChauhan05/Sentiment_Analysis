# Financial News Sentiment Analysis & Stock Movement Prediction API

A FastAPI-based machine learning application that analyzes financial news sentiment using FinBERT and evaluates whether sentiment-driven predictions align with actual stock market movements.

## Overview

This project combines Natural Language Processing (NLP) and financial market data to predict stock price direction from news headlines. Financial news articles are collected through NewsAPI, analyzed using FinBERT, aggregated into daily sentiment scores, and compared against real market performance obtained from Yahoo Finance.

The system provides an end-to-end pipeline for sentiment-based stock movement prediction and evaluation.

## Features

* Financial news collection using NewsAPI
* FinBERT-based sentiment analysis
* Confidence-weighted sentiment scoring
* Daily sentiment aggregation
* Historical market data retrieval via Yahoo Finance
* Stock movement prediction (Up, Down, Unchanged)
* Prediction accuracy evaluation
* Confusion matrix generation
* REST API with FastAPI

## Tech Stack

| Category        | Technology                 |
| --------------- | -------------------------- |
| Backend         | FastAPI, Python            |
| NLP Model       | FinBERT (ProsusAI/finbert) |
| ML Framework    | PyTorch, Transformers      |
| Data Processing | Pandas                     |
| Market Data     | Yahoo Finance (yfinance)   |
| News Source     | NewsAPI                    |
| Evaluation      | Scikit-learn               |
| Visualization   | Matplotlib, Seaborn        |

## Project Workflow

```text
NewsAPI
   │
   ▼
Fetch Financial Headlines
   │
   ▼
FinBERT Sentiment Analysis
   │
   ▼
Weighted Sentiment Scoring
   │
   ▼
Daily Sentiment Aggregation
   │
   ▼
Yahoo Finance Market Data
   │
   ▼
Stock Movement Prediction
   │
   ▼
Performance Evaluation
```

## Supported Tickers

* AAPL (Apple)
* TSLA (Tesla)
* NVDA (NVIDIA)
* MSFT (Microsoft)
* AMZN (Amazon)
* GOOGL (Google)
* META (Meta)
* NFLX (Netflix)
* AMD
* JPM (JPMorgan)

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/financial-sentiment-analysis.git

cd financial-sentiment-analysis
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
NEWS_API_KEY=your_newsapi_key
```

## Running the Application

```bash
uvicorn main:app --reload
```

Server URL:

```text
http://127.0.0.1:8000
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```http
GET /
```

Response:

```json
{
  "message": "API is working"
}
```

### Analyze Stock

```http
POST /analyze
```

Request Body:

```json
{
  "ticker": "AMD",
  "start_date": "2026-01-01",
  "end_date": "2026-02-01"
}
```

Example Response:

```json
{
  "accuracy": 72.5,
  "total_articles": 40,
  "correct_predictions": 29,
  "incorrect_predictions": 11,
  "movement_distribution": {
    "up": 15,
    "down": 18,
    "unchanged": 7
  },
  "prediction_distribution": {
    "up": 13,
    "down": 20,
    "unchanged": 7
  },
  "latest_market_data": {
    "open": 145.20,
    "high": 147.80,
    "low": 143.50,
    "close": 146.90,
    "volume": 52300000
  }
}
```

## Sentiment Scoring

| Sentiment | Score |
| --------- | ----- |
| Positive  | +1    |
| Neutral   | 0     |
| Negative  | -1    |

Weighted Score:

```text
Weighted Score = Sentiment Score × Confidence
```

Daily scores are aggregated by ticker and publication date to generate an overall sentiment signal.

## Prediction Logic

```text
Aggregate Score > 0.3   → Up
Aggregate Score < -0.3  → Down
Otherwise               → Unchanged
```

## Evaluation Metrics

The system evaluates prediction performance using:

* Accuracy
* Confusion Matrix
* Correct Predictions
* Incorrect Predictions
* Actual Movement Distribution
* Predicted Movement Distribution

## Future Improvements

* Support additional stock tickers
* Real-time news streaming
* Advanced forecasting models
* Sentiment trend visualization dashboard
* Multi-day movement prediction
* Integration with additional financial news sources

## License

This project is intended for educational and research purposes.
