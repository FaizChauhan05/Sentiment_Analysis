import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from functools import lru_cache


@lru_cache(maxsize=32)
def _get_financial_info(ticker_symbol: str) -> dict:
    try:
        return yf.Ticker(ticker_symbol).info or {}
    except Exception:
        return {}


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


def market_data(df):

    Open = []
    High = []
    Low = []
    Close = []
    Volume = []
    movement = []
    earnings_per_share = []
    revenue_growth = []
    free_cash_flow = []
    net_margins = []
    return_on_equity = []
    pe_list = []
    peg_list = []
    debt_equity_list = []

    def append_financials(fin_info: dict) -> None:
        earnings_per_share.append(_first_number(fin_info, "trailingEps"))
        revenue_growth.append(_first_number(fin_info, "revenueGrowth"))
        free_cash_flow.append(_first_number(fin_info, "freeCashflow", "freeCashFlow"))
        net_margins.append(_first_number(fin_info, "profitMargins"))
        return_on_equity.append(_first_number(fin_info, "returnOnEquity"))
        pe_list.append(_first_number(fin_info, "trailingPE", "forwardPE"))
        peg_list.append(_first_number(fin_info, "pegRatio", "trailingPegRatio"))
        debt_equity_list.append(_first_number(fin_info, "debtToEquity"))

    def get_stock_data(ticker_symbol, start_date, end_date):

        ticker_obj = yf.Ticker(ticker_symbol)
        stock_data = ticker_obj.history(start = start_date, end = end_date, interval = '1d')
        fin_info = _get_financial_info(ticker_symbol)

        if stock_data.empty:

            Open.append("No data")
            High.append("No data")
            Low.append("No data")
            Close.append("No data")
            Volume.append("No data")
            movement.append("No data")
            append_financials(fin_info)
        else:

            olhcv_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

            Open.append(olhcv_data['Open'].iloc[-1])
            High.append(olhcv_data['High'].iloc[-1])
            Low.append(olhcv_data['Low'].iloc[-1])
            Close.append(olhcv_data['Close'].iloc[-1])
            Volume.append(olhcv_data['Volume'].iloc[-1])

            if len(olhcv_data) >= 2:
                previous_close = olhcv_data['Close'].iloc[-2]
                current_close = olhcv_data['Close'].iloc[-1]
            else:
                movement.append("No data")
                append_financials(fin_info)
                return

            if current_close > previous_close:
                movement.append('Stock price went up')
            elif current_close < previous_close:
                movement.append('Stock price went down')
            else:
                movement.append('Stock price remained unchanged')

            append_financials(fin_info)


    market_close_hour = 16

    df['date'] = pd.to_datetime(df['date'], utc = True)

    for index, row in df.iterrows():

        ticker_symbol = row['ticker']

        pure_date = pd.to_datetime(
            row['date']
        )

        pure_date_est = (
            pure_date.tz_convert('America/New_York')
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

    df['EPS'] = earnings_per_share
    df['Revenue_Growth'] = revenue_growth
    df['Free_Cash_Flow'] = free_cash_flow
    df['Net_Profit_Margin'] = net_margins
    df['ROE'] = return_on_equity

    df['PE_Ratio'] = pe_list
    df['PEG_Ratio'] = peg_list
    df['Debt_to_Equity'] = debt_equity_list

    df = df[df['Movement'] != 'No data'].copy()

    return df
