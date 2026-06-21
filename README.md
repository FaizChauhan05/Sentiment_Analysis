# SentiCore | Financial News Sentiment & Stock Movement Prediction

An analytical machine learning dashboard that aggregates financial news headlines, scores sentiment via FinBERT, and predicts daily stock price movements (`up`, `down`, `unchanged`) using a trained 3-class XGBoost model.

[![Streamlit App](https://static.streamlit.io/badge_github_white.svg)](https://sentiment-analysis-finbertt.streamlit.app/)

---

## Key Features

- **Interactive Streamlit Dashboard**: Real-time sentiment metrics, per-class model evaluation reports, transaction/mention feeds, and interactive Plotly visualization charts.
- **Trained 3-Class XGBoost Model**: Predicts market direction based on **14 engineered features** (6 aggregated daily sentiment indicators + 8 corporate financial metrics).
- **Multi-Source RSS Aggregator**: Deduplicates and fetches news from Google News RSS, Yahoo Finance RSS, and Bing News RSS feeds. GDELT API has been removed to avoid rate limits and latency.
- **Automatic Fallback Check**: Normalizes predictions and implements a class-collapse check (falls back to aggregate scoring rules if model predictions collapse to a single direction).

---

## Tech Stack

- **Dashboard**: Streamlit, Plotly
- **NLP / ML Frameworks**: FinBERT (`ProsusAI/finbert` via Hugging Face), XGBoost, Scikit-learn, PyTorch
- **APIs & Data**: Yahoo Finance (`yfinance` for history and corporate metrics)
- **Data Engineering**: Pandas, NumPy

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/FaizChauhan05/Sentiment_Analysis.git
   cd Sentiment_Analysis
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # macOS/Linux:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Launch the Streamlit web interface locally:
```bash
streamlit run streamlit_app.py
```
View the dashboard in your browser at `http://localhost:8501`.

---

## Model Features (14 Input Features)

1. **Daily Sentiment Features (6)**: `Aggregate_Score` (mean of weighted sentiment scores), `Positive_Ratio`, `Negative_Ratio`, `Sentiment_Spread`, `Max_Confidence`, `Article_Count`.
2. **Corporate Financial Features (8)**: `EPS`, `Revenue_Growth`, `Free_Cash_Flow`, `Net_Profit_Margin`, `ROE`, `PE_Ratio`, `PEG_Ratio`, `Debt_to_Equity`.

---

## Disclaimer
This project is for educational and research purposes. Stock predictions and sentiment scores are not financial advice.
