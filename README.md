# SentiCore | Financial News Sentiment Analysis & Stock Movement Prediction

A dual-mode machine learning application featuring an interactive Streamlit analytics dashboard and a FastAPI backend. It analyzes financial news headlines using FinBERT and evaluates whether sentiment-driven predictions align with actual stock market movements.

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
This project combines Natural Language Processing (NLP) and financial market data to predict stock price direction from news headlines. Financial news articles are collected through NewsAPI, analyzed using FinBERT, aggregated into daily sentiment scores, and compared against real market performance obtained from Yahoo Finance.

It offers:
1. An **interactive Streamlit Web Dashboard** (SentiCore) to visualize prediction stats, sentiment trends, news mentions, and correlation reports.
2. A **FastAPI REST API** to execute the pipeline programmatically.

---

## Key Features

### Interactive Dashboard (SentiCore)
The Streamlit application includes four main analytical views:
* **Overview Dashboard**: Displays key metric cards (Accuracy, Total Articles, Correct Predictions, Incorrect Predictions) and lists trending terms/phrases.
* **Analysis Detail**: Displays a granular data table of fetched headlines, source outlets, model confidence levels, and sentiment labels (with optimized light theme contrast).
* **Mentions Feed**: Explores headline distributions, sentiment breakdown, and trending keywords.
* **Reports Page**: Provides dynamic Plotly-powered charts:
  - **Sentiment Over Time**: Line chart showing aggregate daily sentiment compared with daily stock prices.
  - **Confusion Matrix**: Visual heatmap evaluating predictions vs. actual stock movements.
  - **Movement Distribution**: Bar charts showing predictions vs. actual stock price movement counts.
* **Floating Sidebar Controls**: Date/ticker selection inputs with a persistent, non-softlocking sidebar toggle.

### REST API (FastAPI)
* Dynamic news collection from NewsAPI.
* FinBERT-based sentiment analysis and scoring.
* Evaluates accuracy metrics and confusion matrices.
* Returns structured JSON responses.

### Keyword Extraction
* Custom NLP utility that extracts trending keywords (unigrams) and phrases (bigrams) using stop-word filtering to highlight the most discussed terms in the analysis window.

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
│   NewsAPI    │    │Yahoo Finance│
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
* **News Source**: NewsAPI (`newsapi-python`)
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
* **NewsAPI Constraints**: Due to NewsAPI Free Tier limitations, article fetches are restricted to headlines published within a **30-day window** from today.
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

### Configure Environment Variables
Create a `.env` file in the root directory:
```env
NEWS_API_KEY=your_newsapi_key_here
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
  "ticker": "AMD",
  "start_date": "2026-05-15",
  "end_date": "2026-06-12"
}
```
* **Response**:
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

The predicted signal is compared against the actual daily stock close price movement (Yahoo Finance).

---

## License
This project is intended for educational and research purposes.
