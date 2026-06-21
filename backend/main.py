from fastapi import FastAPI, HTTPException
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
    try:
        news_df = fetch_gdelt_news(
            data.ticker,
            data.start_date,
            data.end_date
        )
        if news_df.empty:
            raise HTTPException(
                status_code=404,
                detail="No news articles found for selected ticker/date range",
            )

        sentiment_df = sentiment_analysis(news_df)

        aggregated_df = aggregate_data(sentiment_df)
        market_df = market_data(aggregated_df)
        if market_df.empty:
            raise HTTPException(
                status_code=502,
                detail="Market data could not be retrieved for the selected period",
            )

        normalized_df = normalize_labels(market_df)
        return evaluate_predictions(normalized_df)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
