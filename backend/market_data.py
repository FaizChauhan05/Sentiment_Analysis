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
    earnings_per_share = []
    revenue_growth = []
    free_cash_flow = []
    net_margins = []
    return_on_equity = []
    pe_list = []
    peg_list = []
    debt_equity_list = []
    def get_stock_data(ticker_symbol, start_date, end_date):

        ticker_obj = yf.Ticker(ticker_symbol)
        stock_data = ticker_obj.history(start = start_date, end = end_date, interval = '1d')

        if stock_data.empty:

            Open.append("No data")
            High.append("No data")
            Low.append("No data")
            Close.append("No data")
            Volume.append("No data")
            movement.append("No data")
            earnings_per_share.append(0.0)
            revenue_growth.append(0.0)
            free_cash_flow.append(0.0)
            net_margins.append(0.0)
            return_on_equity.append(0.0)
            pe_list.append(0.0)
            peg_list.append(0.0)
            debt_equity_list.append(0.0)
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
                earnings_per_share.append(0.0)
                revenue_growth.append(0.0)
                free_cash_flow.append(0.0)
                net_margins.append(0.0)
                return_on_equity.append(0.0)
                pe_list.append(0.0)
                peg_list.append(0.0)
                debt_equity_list.append(0.0)
                return

            if current_close > previous_close:
                movement.append('Stock price went up')
            elif current_close < previous_close:
                movement.append('Stock price went down')
            else:
                movement.append('Stock price remained unchanged')

            fin_info = ticker_obj.info

            earnings_per_share.append(fin_info.get('trailingEps', 0.0))
            revenue_growth.append(fin_info.get('revenueGrowth', 0.0))
            free_cash_flow.append(fin_info.get('freeCashflow', 0))
            net_margins.append(fin_info.get('profitMargins', 0.0))
            return_on_equity.append(fin_info.get('returnOnEquity', 0.0))
            pe_list.append(fin_info.get('trailingPE', 0.0))
            peg_list.append(fin_info.get('pegRatio', 0.0))
            debt_equity_list.append(fin_info.get('debtToEquity', 0.0))


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
