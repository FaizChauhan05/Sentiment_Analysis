import pandas as pd
from gdeltdoc import GdeltDoc, Filters

def fetch_gdelt_news(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical news volume using the GDELT 3-month rolling API.
    Formatted to drop seamlessly into the SentiCore pipeline.
    """
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

    
    gd = GdeltDoc()

    
    f = Filters(
        keyword=company,
        start_date=start_date,
        end_date=end_date
    )

    try:
        articles = gd.article_search(f)
        
        if articles.empty:
            return pd.DataFrame()

       
        df = articles.rename(columns={
            'title': 'headline',
            'seendate': 'published_at',
            'domain': 'source'
        })

        
        df['published_at'] = pd.to_datetime(df['published_at']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        df['ticker'] = ticker
        df['company'] = company

        df = df[['ticker', 'company', 'headline', 'source', 'published_at', 'url']]

    
        df = df.drop_duplicates(subset=['headline'])
        
        df = df.sort_values(by='published_at', ascending=False).head(150)

        return df

    except Exception as e:
        print(f"GDELT API Error: {e}")
        return pd.DataFrame()