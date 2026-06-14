from fastapi import FastAPI
from pydantic import BaseModel
from backend.gdelt_fetcher import fetch_gdelt_news
from backend.sentiment import sentiment_analysis
from backend.market_data import market_data
from backend.normalize_labels import normalize_labels
from backend.Evaluation import evaluate_predictions
from backend.aggregation import aggregate_data

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is working"}

class AnalyzeRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

@app.post("/analyze")
def analyze_stock(data: AnalyzeRequest):

    news_df = fetch_gdelt_news(
        data.ticker,
        data.start_date,
        data.end_date
    )
    if news_df.empty:
     return {
        "error": "No news articles found for selected ticker/date range"
        }


    sentiment_df = sentiment_analysis(news_df)

    aggregated_df = aggregate_data(sentiment_df)
   
    market_df = market_data(aggregated_df)

    normalized_df = normalize_labels(market_df)

    results = evaluate_predictions(normalized_df)

    return results