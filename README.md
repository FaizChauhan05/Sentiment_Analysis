# SentiCore | Financial News Sentiment Analysis & Stock Movement Prediction https://sentiment-analysis-finbertt.streamlit.app/

A dual-mode machine learning application featuring an interactive Streamlit analytics dashboard and a FastAPI backend. It analyzes financial news headlines using FinBERT and evaluates whether sentiment-driven predictions align with actual stock market movements.

> [!WARNING]
> **Known Issue / Notice:** The current threshold-based prediction mechanism correlating aggregated sentiment scores to actual stock movement is hardcoded (threshold set at `<-0.3` for bearish and `>0.3` for bullish). This logic is temporary and will be updated soon with a more advanced model.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
   - [Interactive Dashboard (SentiCore)](#interactive-dashboard-senticore)
   - [REST API (FastAPI)](#rest-api-fastapi)
3. [Architecture & Workflow](#architecture--workflow)
4. [Tech Stack](#tech-stack)
5. [Supported Tickers](#supported-tickers)
6. [System Limitations & Disclaimers](#system-limitations--disclaimers)
7. [Installation & Setup](#installation--setup)
8. [Running the Application](#running-the-application)
9. [API Endpoints](#api-endpoints)
10. [Sentiment Scoring & Prediction Logic](#sentiment-scoring--prediction-logic)
11. [License](#license)

---

## Overview
This project combines Natural Language Processing (NLP) and financial market data to predict stock price direction from news headlines. Financial news articles are collected through a multi-source news aggregator (GDELT 2.0 DOC API, Yahoo Finance, Google News, and Bing News), analyzed using a pre-trained FinBERT model, aggregated into daily sentiment scores, and correlated with real market performance obtained from Yahoo Finance.

It offers:
1. An **interactive Streamlit Web Dashboard** (SentiCore) to visualize prediction stats, sentiment trends, news mentions, and correlation reports.
2. A **FastAPI REST API** to execute the pipeline programmatically.

---

## Key Features

### Interactive Dashboard (SentiCore)
The Streamlit application includes four main analytical views:
* **Overview Dashboard**: Displays key metric cards (Accuracy, Total Articles, Correct Predictions, Last Close Price) and lists trending terms/phrases.
* **Analysis Detail**: Displays a breakdown of positive, neutral, and negative sentiment distribution, per-class model accuracy (Precision, Recall, F1), keyword cloud, and AI sentiment summary.
* **Mentions Feed**: Explores article-level headlines, outlets, dates, sentiment labels, and confidence metrics with full search and filtering controls.
* **Reports Page**: Provides dynamic Plotly-powered charts:
  - **Sentiment Over Time**: Line chart showing aggregate daily sentiment compared with daily stock prices.
  - **Confusion Matrix**: Visual heatmap evaluating predictions vs. actual stock movements.
  - **Movement Distribution**: Bar charts showing predictions vs. actual stock price movement counts.
  - **Latest Market Data (OHLCV)**: Clean, high-contrast display of Open, High, Low, Close, and Volume.
* **Floating Sidebar Controls**: Date/ticker selection inputs with a persistent, non-softlocking sidebar toggle.

### REST API (FastAPI)
* Dynamic news collection via multi-source aggregator (GDELT + RSS).
* FinBERT-based sentiment analysis and scoring.
* Evaluates accuracy metrics and confusion matrices.
* Returns structured JSON responses.

### Intelligent Multi-Source Fetcher
* **Date-Chunking RSS Engine**: Bypasses RSS pagination constraints by splitting requests into 30-day chunks (e.g., yielding 1,100+ unique articles over a 90-day window).
* **Graceful Failover**: Uses exponential backoff with jitter for GDELT API requests and automatically falls back to Yahoo Finance, Google News, and Bing News RSS streams if GDELT is rate-limited.
* **Deduplication**: Automatically filters out duplicate headlines based on title text and publication source.

---

## Architecture & Workflow

```text
       ┌──────────────────────┐
       │   Streamlit App      │
       │  (Frontend UI)       │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │   FastAPI Backend    │
       │    (REST API)        │
       └──────────┬───────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│GDELT / RSS   │    │Yahoo Finance│
│(Headlines)   │    │ (Market Data)│
└───────┬──────┘    └───────┬──────┘
        │                   │
        ▼                   │
┌──────────────┐            │
│   FinBERT    │            │
│(Sentiment)   │            │
└───────┬──────┘            │
        │                   │
        ▼                   ▼
┌───────────────────────────┐
│    Sentiment Scoring      │
│  & Stock Prediction Logic │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│  Performance Evaluation   │
│ & Visualizations (Plotly) │
└───────────────────────────┘
```

---

## Tech Stack
* **Frontend Dashboard**: Streamlit, Plotly
* **Backend API**: FastAPI, Uvicorn
* **NLP Model**: FinBERT (`ProsusAI/finbert` via Hugging Face Transformers)
* **ML Framework**: PyTorch
* **Data Processing**: Pandas, NumPy
* **Market Data**: Yahoo Finance (`yfinance`)
* **News Sources**: GDELT 2.0 DOC API, Google News RSS, Yahoo Finance RSS, Bing News RSS (custom standard-library XML parsing)
* **Evaluation**: Scikit-learn

---

## Supported Tickers
* AAPL (Apple)
* TSLA (Tesla)
* NVDA (NVIDIA)
* MSFT (Microsoft)
* AMZN (Amazon)
* GOOGL (Google)
* META (Meta)
* NFLX (Netflix)
* AMD (Advanced Micro Devices)
* JPM (JPMorgan Chase)

---

## System Limitations & Disclaimers
* **Historical Window**: News collection fetches headlines published within a **90-day window** from today.
* **No API Keys Required**: Bypasses the strict 30-day restriction and API key requirement of NewsAPI by using public APIs and RSS feeds.
* **Model Constraints**: Because of local hosting costs and model parameter size, FinBERT prediction accuracy and processing latency may vary depending on the total volume of articles collected.
* **Disclaimer**: This tool is for educational purposes only. Sentiment predictions are not financial advice.

---

## Installation & Setup

### Clone Repository
```bash
git clone https://github.com/FaizChauhan05/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### Create Virtual Environment
```bash
python -m venv venv
```
Activate it:
* **Windows**: `venv\Scripts\activate`
* **macOS/Linux**: `source venv/bin/activate`

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1. Run the FastAPI Backend
Start the FastAPI server:
```bash
uvicorn backend.main:app --reload
```
* API Documentation: http://127.0.0.1:8000/docs
* Health Check: http://127.0.0.1:8000/

### 2. Run the Streamlit Dashboard
Launch the interactive web UI:
```bash
streamlit run streamlit_app.py
```
* Dashboard URL: http://localhost:8501

---

## API Endpoints

### Health Check
`GET /`
* **Response**: `{"message": "API is working"}`

### Analyze Ticker
`POST /analyze`
* **Request Body**:
```json
{
  "ticker": "AAPL",
  "start_date": "2026-03-15",
  "end_date": "2026-06-12"
}
```
* **Response**:
```json
{
  "accuracy": 64.77,
  "total_articles": 88,
  "correct_predictions": 57,
  "incorrect_predictions": 31,
  "movement_distribution": {
    "up": 44,
    "down": 36,
    "unchanged": 8
  },
  "prediction_distribution": {
    "up": 41,
    "down": 39,
    "unchanged": 8
  },
  "latest_market_data": {
    "open": 289.45,
    "high": 293.10,
    "low": 288.75,
    "close": 291.13,
    "volume": 49200000
  }
}
```

---

## Sentiment Scoring & Prediction Logic

### Scoring System
| Sentiment | Score |
| --------- | ----- |
| Positive  | +1    |
| Neutral   | 0     |
| Negative  | -1    |

Weighted Score calculation:
$$\text{Weighted Score} = \text{Sentiment Score} \times \text{Confidence}$$

Daily scores are aggregated by ticker and publication date.

### Prediction Logic
* **Aggregate Daily Score > 0.3**  $\rightarrow$ **Up**
* **Aggregate Daily Score < -0.3** $\rightarrow$ **Down**
* **Otherwise**                    $\rightarrow$ **Unchanged**

The predicted signal is compared against the actual daily stock close price movement (Yahoo Finance). This will change soon depending upon what I think is best

---

## License
This project is intended for educational and research purposes.
