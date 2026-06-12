import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay


def market_data(df):

    Open = []
    High = []
    Low = []
    Close = []
    Volume = []
    movement = []

    def get_stock_data(
        ticker_symbol,
        start_date,
        end_date
    ):

        stock_data = yf.Ticker(
            ticker_symbol
        ).history(
            start=start_date,
            end=end_date,
            interval='1d'
        )
        if stock_data.empty:

            Open.append("No data")
            High.append("No data")
            Low.append("No data")
            Close.append("No data")
            Volume.append("No data")
            movement.append("No data")

        else:

            olhcv_data = stock_data[
                ['Open', 'High', 'Low', 'Close', 'Volume']
            ]

            # Append latest available OHLCV (use last row)
            Open.append(olhcv_data['Open'].iloc[-1])
            High.append(olhcv_data['High'].iloc[-1])
            Low.append(olhcv_data['Low'].iloc[-1])
            Close.append(olhcv_data['Close'].iloc[-1])
            Volume.append(olhcv_data['Volume'].iloc[-1])

            # Determine movement comparing last two closes if possible
            if len(olhcv_data) >= 2:
                previous_close = olhcv_data['Close'].iloc[-2]
                current_close = olhcv_data['Close'].iloc[-1]
            else:
                movement.append("No data")
                return

            if current_close > previous_close:
                movement.append('Stock price went up')
            elif current_close < previous_close:
                movement.append('Stock price went down')
            else:
                movement.append('Stock price remained unchanged')
    
            

    market_close_hour = 16

    
    df['date'] = pd.to_datetime(
        df['date'],
        utc=True
    )

    for index, row in df.iterrows():

        ticker_symbol = row['ticker']

        pure_date = pd.to_datetime(
            row['date']
        )

        pure_date_est = (
            pure_date.tz_convert('US/Eastern')
            if pure_date.tzinfo
            else pure_date
        )

        if pure_date_est.hour >= market_close_hour:

            target_date = pure_date_est + BDay(1)

        else:

            target_date = pure_date_est

        start_date = (
        target_date - BDay(2)
        ).strftime('%Y-%m-%d')

        end_date = (
            target_date + BDay(1)
        ).strftime('%Y-%m-%d')

        get_stock_data(
            ticker_symbol,
            start_date,
            end_date
        )

    df['Open'] = Open
    df['High'] = High
    df['Low'] = Low
    df['Close'] = Close
    df['Volume'] = Volume
    df['Movement'] = movement

    df = df[df['Movement'] != 'No data'].copy()

    return df