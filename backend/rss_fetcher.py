"""Multi-source financial news RSS fetcher for SentiCore."""

import random
import re
import time
from datetime import timedelta
from xml.etree import ElementTree

import pandas as pd
import requests

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GOOGLE_NEWS_TIMEOUT = 15
GOOGLE_NEWS_CHUNK_DAYS = 30

YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
YAHOO_TIMEOUT = 15

BING_NEWS_RSS = "https://www.bing.com/news/search"
BING_NEWS_TIMEOUT = 15

COMPANY_MAPPING: dict[str, str] = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "GOOG": "Google",
    "META": "Meta",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "JPM": "JPMorgan",
    "INTC": "Intel",
    "BA": "Boeing",
    "DIS": "Disney",
    "V": "Visa",
}

_FINANCIAL_QUERY_SUFFIXES = ("stock", "earnings", "shares", "revenue")
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def _clamp_window(
    start_date: str, end_date: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("start_date and end_date must be valid date strings")
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")
    if (end_dt - start_dt).days > 90:
        start_dt = end_dt - timedelta(days=90)

    return start_dt.normalize(), end_dt.normalize()


def _fetch_google_news_query(query_str: str) -> list[dict]:
    params = {
        "q": query_str,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        resp = requests.get(
            GOOGLE_NEWS_RSS, params=params, headers=headers, timeout=GOOGLE_NEWS_TIMEOUT
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Google News] Error for '{query_str}': {exc}")
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as exc:
        print(f"[Google News] XML parse error: {exc}")
        return []

    articles: list[dict] = []
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

        articles.append(
            {
                "title": title,
                "url": link,
                "seendate": pub_date,
                "domain": source,
            }
        )

    return articles


def _date_chunks(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    chunk_days: int = GOOGLE_NEWS_CHUNK_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _fetch_from_google_news(
    ticker: str,
    company: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> list[dict]:
    search_terms: list[str] = [ticker]
    if company.upper() != ticker.upper():
        search_terms.append(company)
        for suffix in _FINANCIAL_QUERY_SUFFIXES:
            search_terms.append(f"{company} {suffix}")

    chunks = _date_chunks(start_dt, end_dt)
    all_articles: list[dict] = []
    query_count = 0

    for chunk_start, chunk_end in chunks:
        date_filter = (
            f"after:{chunk_start.strftime('%Y-%m-%d')} "
            f"before:{chunk_end.strftime('%Y-%m-%d')}"
        )
        for term in search_terms:
            query_count += 1
            arts = _fetch_google_news_query(f"{term} {date_filter}")
            all_articles.extend(arts)
            time.sleep(0.3)

    print(
        f"[Google News] {len(all_articles)} articles from {query_count} queries "
        f"({len(chunks)} date chunks × {len(search_terms)} terms)"
    )
    return all_articles


def _fetch_from_yahoo_rss(ticker: str) -> list[dict]:
    params = {"s": ticker, "region": "US", "lang": "en-US"}
    headers = {"User-Agent": _USER_AGENT}

    try:
        resp = requests.get(
            YAHOO_RSS_URL, params=params, headers=headers, timeout=YAHOO_TIMEOUT
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[Yahoo RSS] Error: {exc}")
        return []

    try:
        root = ElementTree.fromstring(resp.content)
    except ElementTree.ParseError as exc:
        print(f"[Yahoo RSS] XML parse error: {exc}")
        return []

    articles: list[dict] = []
    for item in root.iter("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source = item.findtext("source", "Yahoo Finance").strip()

        if not title:
            continue

        articles.append(
            {
                "title": title,
                "url": link,
                "seendate": pub_date,
                "domain": source,
            }
        )

    print(f"[Yahoo RSS] {len(articles)} articles")
    return articles


def _fetch_from_bing_news(
    ticker: str,
    company: str,
) -> list[dict]:
    search_terms: list[str] = [f"{ticker} stock"]
    if company.upper() != ticker.upper():
        search_terms.extend(
            [
                f"{company} stock",
                f"{company} earnings",
                f"{company} shares",
            ]
        )

    headers = {"User-Agent": _USER_AGENT}
    all_articles: list[dict] = []

    for term in search_terms:
        params = {"q": term, "format": "rss"}
        try:
            resp = requests.get(
                BING_NEWS_RSS,
                params=params,
                headers=headers,
                timeout=BING_NEWS_TIMEOUT,
            )
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

            all_articles.append(
                {
                    "title": title,
                    "url": link,
                    "seendate": pub_date,
                    "domain": source,
                }
            )

        time.sleep(0.3)

    print(f"[Bing News] {len(all_articles)} articles from {len(search_terms)} queries")
    return all_articles


def _normalise_headline(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _normalise_articles(
    raw_articles: list[dict],
    ticker: str,
    company: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []
    for art in raw_articles:
        headline = art.get("title", "").strip()
        published = art.get("seendate", "")

        if not headline or not published:
            continue

        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "headline": headline,
                "source": art.get("domain", ""),
                "published_at": published,
                "url": art.get("url", ""),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])

    start_utc = pd.Timestamp(start_dt, tz="UTC")
    end_utc = (
        pd.Timestamp(end_dt, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )
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

    all_articles: list[dict] = []

    print(f"[Fetcher] Fetching from Google News for {ticker}…")
    all_articles.extend(_fetch_from_google_news(ticker, company, start_dt, end_dt))

    print(f"[Fetcher] Fetching from Yahoo Finance RSS for {ticker}…")
    all_articles.extend(_fetch_from_yahoo_rss(ticker))

    print(f"[Fetcher] Fetching from Bing News for {ticker}…")
    all_articles.extend(_fetch_from_bing_news(ticker, company))

    print(f"[Fetcher] Total raw articles (all sources): {len(all_articles)}")
    df = _normalise_articles(all_articles, ticker, company, start_dt, end_dt)
    print(f"[Fetcher] After dedup & date filter: {len(df)} unique articles for {ticker}")

    return df
