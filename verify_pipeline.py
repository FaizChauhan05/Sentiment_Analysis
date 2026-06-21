#!/usr/bin/env python3
"""
verify_pipeline.py
Locally verifies the news fetching, yfinance historical alignment, sentiment aggregation, 
and feature engineering logic for the SentiCore XGBoost retraining pipeline.
"""

import sys
import os
import re
import time
import random
import threading
from datetime import datetime, timedelta
from xml.etree import ElementTree
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from pandas.tseries.offsets import BDay

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GOOGLE_NEWS_TIMEOUT = 15
GOOGLE_NEWS_CHUNK_DAYS = 30

YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
YAHOO_TIMEOUT = 15

BING_NEWS_RSS = "https://www.bing.com/news/search"
BING_NEWS_TIMEOUT = 15

_FINANCIAL_QUERY_SUFFIXES = ("stock", "earnings", "shares", "revenue")
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

COMPANY_MAPPING = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "META": "Meta",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "JPM": "JPMorgan",
}

def _clamp_window(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("start_date and end_date must be valid date strings")
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")
    return start_dt.normalize(), end_dt.normalize()

def _fetch_google_news_query(query_str: str) -> list[dict]:
    params = {"q": query_str, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    headers = {"User-Agent": _USER_AGENT}
    try:
        resp = requests.get(GOOGLE_NEWS_RSS, params=params, headers=headers, timeout=GOOGLE_NEWS_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Google News] Error for '{query_str}': {exc}")
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as exc:
        print(f"[Google News] XML parse error: {exc}")
        return []

    articles = []
    for item in root.iter("item"):
        title_raw = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source_elem = item.findtext("source", "").strip()

        if not title_raw:
            continue

        title = title_raw
        source = source_elem or "Google News"
        if " - " in title_raw:
            parts = title_raw.rsplit(" - ", 1)
            title = parts[0].strip()
            if not source_elem:
                source = parts[1].strip()

        articles.append({
            "title": title,
            "url": link,
            "seendate": pub_date,
            "domain": source,
        })
    return articles

def _date_chunks(start_dt: pd.Timestamp, end_dt: pd.Timestamp, chunk_days: int = GOOGLE_NEWS_CHUNK_DAYS):
    chunks = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks

def _fetch_from_google_news(ticker: str, company: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> list[dict]:
    if company.upper() != ticker.upper():
        query_str = f'("{ticker}" OR "{company}" OR "{company} stock" OR "{company} earnings" OR "{company} shares" OR "{company} revenue")'
    else:
        query_str = f'"{ticker}"'

    chunks = _date_chunks(start_dt, end_dt)
    all_articles = []
    for chunk_start, chunk_end in chunks:
        date_filter = f"after:{chunk_start.strftime('%Y-%m-%d')} before:{chunk_end.strftime('%Y-%m-%d')}"
        arts = _fetch_google_news_query(f"{query_str} {date_filter}")
        all_articles.extend(arts)
        time.sleep(0.5)
    return all_articles

def _fetch_from_yahoo_rss(ticker: str) -> list[dict]:
    params = {"s": ticker, "region": "US", "lang": "en-US"}
    headers = {"User-Agent": _USER_AGENT}
    try:
        resp = requests.get(YAHOO_RSS_URL, params=params, headers=headers, timeout=YAHOO_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Yahoo RSS] Error: {exc}")
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as exc:
        print(f"[Yahoo RSS] XML parse error: {exc}")
        return []

    articles = []
    for item in root.iter("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source = item.findtext("source", "Yahoo Finance").strip()

        if not title:
            continue

        articles.append({
            "title": title,
            "url": link,
            "seendate": pub_date,
            "domain": source,
        })
    return articles

def _fetch_from_bing_news(ticker: str, company: str) -> list[dict]:
    search_terms = [f"{ticker} stock"]
    if company.upper() != ticker.upper():
        search_terms.extend([
            f"{company} stock",
            f"{company} earnings",
            f"{company} shares",
        ])

    headers = {"User-Agent": _USER_AGENT}
    all_articles = []
    for term in search_terms:
        params = {"q": term, "format": "rss"}
        try:
            resp = requests.get(BING_NEWS_RSS, params=params, headers=headers, timeout=BING_NEWS_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[Bing News] Error for '{term}': {exc}")
            continue

        try:
            root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError:
            continue

        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            pub_date = item.findtext("pubDate", "").strip()

            if not title:
                continue

            source = "Bing News"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip()

            all_articles.append({
                "title": title,
                "url": link,
                "seendate": pub_date,
                "domain": source,
            })
        time.sleep(0.3)
    return all_articles

def _normalise_headline(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

def _normalise_articles(raw_articles: list[dict], ticker: str, company: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for art in raw_articles:
        headline = art.get("title", "").strip()
        published = art.get("seendate", "")
        if not headline or not published:
            continue

        rows.append({
            "ticker": ticker,
            "company": company,
            "headline": headline,
            "source": art.get("domain", ""),
            "published_at": published,
            "url": art.get("url", ""),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])

    start_utc = pd.Timestamp(start_dt, tz="UTC")
    end_utc = pd.Timestamp(end_dt, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df[df["published_at"].between(start_utc, end_utc)]

    df["_norm"] = df["headline"].apply(_normalise_headline)
    df = df.drop_duplicates(subset=["_norm"])
    df = df.drop(columns=["_norm"])

    df = df.sort_values("published_at", ascending=False)
    df["published_at"] = df["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return df.reset_index(drop=True)

def fetch_rss_news(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker = ticker.upper().strip()
    company = COMPANY_MAPPING.get(ticker, ticker)
    start_dt, end_dt = _clamp_window(start_date, end_date)

    all_articles = []
    print(f"[Fetcher] Fetching from Google News for {ticker}...")
    all_articles.extend(_fetch_from_google_news(ticker, company, start_dt, end_dt))

    print(f"[Fetcher] Fetching from Yahoo Finance RSS for {ticker}...")
    all_articles.extend(_fetch_from_yahoo_rss(ticker))

    print(f"[Fetcher] Fetching from Bing News for {ticker}...")
    all_articles.extend(_fetch_from_bing_news(ticker, company))

    print(f"[Fetcher] Total raw articles (all sources): {len(all_articles)}")
    df = _normalise_articles(all_articles, ticker, company, start_dt, end_dt)
    print(f"[Fetcher] After dedup & date filter: {len(df)} unique articles for {ticker}")
    return df

def mock_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Mock sentiment scoring for local testing to avoid 1GB model download."""
    if df.empty or "headline" not in df.columns:
        return df
    
    sentiments = []
    confidences = []
    for h in df["headline"]:
        h_lower = h.lower()
        if any(w in h_lower for w in ["gain", "rise", "up", "high", "surpass", "profit", "growth", "bull", "upgrade"]):
            sentiment = "positive"
        elif any(w in h_lower for w in ["fall", "drop", "down", "low", "loss", "decline", "bear", "downgrade", "debt"]):
            sentiment = "negative"
        else:
            sentiment = "neutral"
        sentiments.append(sentiment)
        confidences.append(random.uniform(0.6, 0.99))
        
    df["Sentiment"] = sentiments
    df["Confidence"] = confidences
    return df



def _first_number(info: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        try:
            if pd.notna(value):
                return float(value)
        except (TypeError, ValueError):
            continue
    return default

def fetch_ticker_financials(ticker_symbol: str) -> dict:
    print(f"[Financials] Fetching yfinance info for {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info or {}
    except Exception as e:
        print(f"[Financials] Error fetching info for {ticker_symbol}: {e}")
        info = {}
        
    return {
        "EPS": _first_number(info, "trailingEps"),
        "Revenue_Growth": _first_number(info, "revenueGrowth"),
        "Free_Cash_Flow": _first_number(info, "freeCashflow", "freeCashFlow"),
        "Net_Profit_Margin": _first_number(info, "profitMargins"),
        "ROE": _first_number(info, "returnOnEquity"),
        "PE_Ratio": _first_number(info, "trailingPE", "forwardPE"),
        "PEG_Ratio": _first_number(info, "pegRatio", "trailingPegRatio"),
        "Debt_to_Equity": _first_number(info, "debtToEquity")
    }

def fetch_ticker_data(ticker: str, start_date: str, end_date: str) -> tuple[str, dict, pd.DataFrame]:
    financials = fetch_ticker_financials(ticker)
    try:
        ticker_obj = yf.Ticker(ticker)
        start_pad = (pd.to_datetime(start_date) - BDay(5)).strftime("%Y-%m-%d")
        end_pad = (pd.to_datetime(end_date) + BDay(5)).strftime("%Y-%m-%d")
        print(f"[Stock History] Downloading history for {ticker} from {start_pad} to {end_pad}...")
        history = ticker_obj.history(start=start_pad, end=end_pad, interval="1d")
        history.index = history.index.tz_localize(None)
    except Exception as e:
        print(f"[Stock History] Error fetching history for {ticker}: {e}")
        history = pd.DataFrame()
    return ticker, financials, history


def aggregate_sentiment_features(articles_df: pd.DataFrame) -> pd.DataFrame:
    if articles_df.empty:
        return pd.DataFrame()
    
    df = articles_df.copy()
    scores = []
    weighted_scores = []
    for _, row in df.iterrows():
        sentiment = str(row['Sentiment']).lower()
        confidence = float(row['Confidence'])
        if sentiment == 'positive':
            score = 1.0
        elif sentiment == 'negative':
            score = -1.0
        else:
            score = 0.0
        scores.append(score)
        weighted_scores.append(score * confidence)
        
    df['score'] = scores
    df['weighted_score'] = weighted_scores
    
    df['date'] = pd.to_datetime(df['published_at'], utc=True).dt.normalize()
    
    grouped = df.groupby(['ticker', 'date'])
    
    aggregated = grouped.agg(
        Aggregate_Score=('weighted_score', 'mean'),
        Article_Count=('headline', 'count'),
        Max_Confidence=('Confidence', 'max')
    ).reset_index()
    
    pos_counts = grouped.apply(lambda x: (x['Sentiment'] == 'positive').sum(), include_groups=False)
    neg_counts = grouped.apply(lambda x: (x['Sentiment'] == 'negative').sum(), include_groups=False)
    
    pos_neg = pd.DataFrame({
        'pos_count': pos_counts,
        'neg_count': neg_counts
    }).reset_index()
    
    aggregated = pd.merge(aggregated, pos_neg, on=['ticker', 'date'])
    
    aggregated['Positive_Ratio'] = aggregated['pos_count'] / aggregated['Article_Count']
    aggregated['Negative_Ratio'] = aggregated['neg_count'] / aggregated['Article_Count']
    aggregated['Sentiment_Spread'] = aggregated['Positive_Ratio'] - aggregated['Negative_Ratio']
    
    aggregated = aggregated.drop(columns=['pos_count', 'neg_count'])
    return aggregated

def get_target_date(utc_date) -> pd.Timestamp:
    utc_dt = pd.to_datetime(utc_date, utc=True)
    est_dt = utc_dt.tz_convert('America/New_York')
    if est_dt.hour >= 16:
        target_date = est_dt + BDay(1)
    else:
        target_date = est_dt
    return target_date.normalize()

def get_movement_and_ohlcv(history_df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    target_dt_naive = pd.to_datetime(target_date).tz_localize(None)
    sub_df = history_df[history_df.index <= target_dt_naive]
    if len(sub_df) < 2:
        return {
            "Open": np.nan, "High": np.nan, "Low": np.nan, 
            "Close": np.nan, "Volume": np.nan, "Pct_Change": np.nan, "Movement": "No data"
        }
    
    last_row = sub_df.iloc[-1]
    prev_row = sub_df.iloc[-2]
    
    current_close = float(last_row['Close'])
    previous_close = float(prev_row['Close'])
    
    pct_change = (current_close - previous_close) / previous_close
    
    if pct_change > 0.005:
        movement = 'up'
    elif pct_change < -0.005:
        movement = 'down'
    else:
        movement = 'unchanged'
        
    return {
        "Open": float(last_row['Open']),
        "High": float(last_row['High']),
        "Low": float(last_row['Low']),
        "Close": current_close,
        "Volume": int(last_row['Volume']),
        "Pct_Change": pct_change,
        "Movement": movement
    }

def verify_pipeline():
    print("--- STARTING RETRAIN PIPELINE LOCAL VERIFICATION ---")
    tickers = ["AAPL", "TSLA"]
    start_date = "2026-05-15"
    end_date = "2026-05-17"

    market_cache = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(fetch_ticker_data, t, start_date, end_date) for t in tickers]
        for f in futures:
            ticker, financials, history = f.result()
            market_cache[ticker] = {"financials": financials, "history": history}

    for t in tickers:
        print(f"Ticker {t}: Financials keys: {list(market_cache[t]['financials'].keys())}")
        print(f"Ticker {t}: Stock History shape: {market_cache[t]['history'].shape}")

    news_list = []
    for t in tickers:
        df_news = fetch_rss_news(t, start_date, end_date)
        if not df_news.empty:
            news_list.append(df_news)
            
    if not news_list:
        print("Error: No news articles fetched. Exiting verification.")
        sys.exit(1)
        
    all_news_df = pd.concat(news_list, ignore_index=True)
    print(f"Total articles fetched: {len(all_news_df)}")

    scored_news_df = mock_sentiment_analysis(all_news_df)
    print(f"Scored news shape: {scored_news_df.shape}")

    agg_df = aggregate_sentiment_features(scored_news_df)
    print(f"Aggregated sentiment shape: {agg_df.shape}")
    print(agg_df.head())

    final_rows = []
    for _, row in agg_df.iterrows():
        t = row["ticker"]
        utc_date = row["date"]
        
        target_dt = get_target_date(utc_date)
        cache_data = market_cache.get(t, {})
        history_df = cache_data.get("history", pd.DataFrame())
        financials = cache_data.get("financials", {})
        
        movement_data = get_movement_and_ohlcv(history_df, target_dt)
        if movement_data["Movement"] == "No data":
            continue
            
        combined_row = {
            "ticker": t,
            "date": utc_date.strftime("%Y-%m-%d"),
            **row.to_dict(),
            **financials,
            **movement_data
        }
        final_rows.append(combined_row)

    if not final_rows:
        print("Warning: No aligned rows found. Adjusting dates or mock stock data may be needed.")
        sys.exit(1)
        
    final_df = pd.DataFrame(final_rows)
    print(f"Final aligned dataset shape: {final_df.shape}")
    print("Final features:")
    print(final_df.columns.tolist())

    sentiment_features = ["Aggregate_Score", "Positive_Ratio", "Negative_Ratio", "Sentiment_Spread", "Max_Confidence", "Article_Count"]
    financial_features = ["EPS", "Revenue_Growth", "Free_Cash_Flow", "Net_Profit_Margin", "ROE", "PE_Ratio", "PEG_Ratio", "Debt_to_Equity"]
    all_features = sentiment_features + financial_features
    
    missing_features = [f for f in all_features if f not in final_df.columns]
    if missing_features:
        print(f"Error: Missing target training features: {missing_features}")
        sys.exit(1)
    else:
        print(f"Success! All {len(all_features)} target training features are present.")

    print("\n--- PIPELINE LOCAL VERIFICATION SUCCESSFUL ---")

if __name__ == "__main__":
    verify_pipeline()
