import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

url = "https://newsapi.org/v2/everything"


def fetch_news(ticker, start_date, end_date):

    company_mapping = {
        "AAPL": "Apple",
        "TSLA": "Tesla",
        "NVDA": "Nvidia",
        "MSFT": "Microsoft",
        "AMZN": "Amazon",
        "GOOGL": "Google",
        "META": "Meta",
        "NFLX": "Netflix",
        "AMD": "AMD",
        "JPM": "JPMorgan"
    }

    company = company_mapping.get(ticker)

    if not company:

        return pd.DataFrame()

    query = (
        f'"{company}" OR "{ticker}" '
        f'AND (stock OR shares OR earnings OR revenue OR market OR investors)'
    )

    params = {
        "q": query,
        "searchIn": "title",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "from": start_date,
        "to": end_date,
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()
    if data.get("status") == "error":
        raise Exception(data.get("message"))

    articles = data.get("articles", [])

    news_data = []

    for article in articles:

        headline = article.get("title")

        if headline:

            news_data.append({
                "ticker": ticker,
                "company": company,
                "headline": headline,
                "source": article.get("source", {}).get("name"),
                "published_at": article.get("publishedAt"),
                "url": article.get("url"),
                "description": article.get("description")
            })

    df = pd.DataFrame(news_data)

    df.drop_duplicates(subset=["headline"], inplace=True)

    return df