"""Utilities for master portfolio"""
from enum import Enum
import datetime
import warnings
import numpy as np
import pandas as pd
import pandas.errors
import trading_calendars


BBG_EXCHANGE_MAP = {
    "11": "UN",  # NYSE
    "14": {
        "Q": "UW",  # Nasdaq Global Select Market
        "G": "UQ",  # Nasdaq Global Market
        "S": "UR",  # Nasdaq Capital Market
        "0": "UR",  # Nasdaq Capital Market
    },
    None: "(Non-invested)"
    # "17": "UP",  # NYSE Arca
}
EX_CODES_MAP = {
    "UN": "11",
    "UW": "14",
    "UR": "14",
    "UQ": "14",
}
POST_BBG_TICKER_ADJUSTMENT_MAP = {
    "DOCU UQ": "DOCU UW",
    "TWST UQ": "TWST UW"
}


def post_adjust_bbg_ticker(bbg_ticker):
    """Post-adjust BBG Ticker."""
    global POST_BBG_TICKER_ADJUSTMENT_MAP

    if bbg_ticker in POST_BBG_TICKER_ADJUSTMENT_MAP:
        return POST_BBG_TICKER_ADJUSTMENT_MAP[bbg_ticker]
    else:
        return bbg_ticker


def get_latest_bbg_ticker(api, gvkey, iid):
    """Transform gvkey, iid to bloomberg ticker."""
    global BBG_EXCHANGE_MAP
    print(gvkey, iid)

    data = api.compustat.read_sql(f"""
    (
        select item, itemvalue
        from sec_history
        where gvkey='{gvkey}' and iid='{iid}' and item='EXCHG' and thrudate is null
        order by effdate desc
        limit 1
    )
    union all
    (
        select item, itemvalue
        from sec_history
        where gvkey='{gvkey}' and iid='{iid}' and item='EXCHGTIER' and thrudate is null
        order by effdate desc
        limit 1
    )
    union all
    (
        select 'TICKER' item, tic
        from cssecurity
        where gvkey='{gvkey}' and iid='{iid}'
        order by effdate desc
        limit 1
    );
    """)
    data = data.set_index("item")["itemvalue"].to_dict()
    ticker = data["TICKER"]
    exchange = data.get("EXCHG", None)
    tier = data.get("EXCHGTIER", None)

    if tier is None:
        bbg_ticker = ticker + " " + BBG_EXCHANGE_MAP[exchange]
    else:
        bbg_ticker = ticker + " " + BBG_EXCHANGE_MAP[exchange][tier]

    return post_adjust_bbg_ticker(bbg_ticker)


def get_gvkey_iid_from_latest_bbg_ticker(api, bbg_ticker):
    """Retrieve gvkey_iid from BBG Ticker."""
    global EX_CODES_MAP

    ticker, exchange = bbg_ticker.strip().split(" ")
    ex_code = EX_CODES_MAP[exchange]

    try:
        data = api.compustat.read_sql(
            f"select gvkey, iid from security where tic='{ticker}' and exchg='{ex_code}';"
        )
        if len(data) == 1:
            data = data.iloc[0]
            gvkey, iid = data["gvkey"], data["iid"]
            return gvkey + "_" + iid
        else:
            for i in range(len(data)):
                row = data.iloc[i]
                print(f"({i}) gvkey: {row['gvkey']}, iid: {row['iid']}")
            n = int(input("Multiple Data is searched. Choose number: ").strip())
            data = data.iloc[n]
            gvkey, iid = data["gvkey"], data["iid"]
            return gvkey + "_" + iid

    except pandas.errors.EmptyDataError:
        print(f"{bbg_ticker} is not in NYSE/NASDAQ. Return BBG Ticker with FAIL tag. Need to Check.")
        return "FAIL-" + bbg_ticker


def get_latest_adv(api, gvkey, iid, n):
    return api.compustat.read_sql(f"""
    select AVG(trade)
    from 
    (
        select prccd * cshtrd as trade
        from sec_dprc 
        where gvkey='{gvkey}' and iid='{iid}'
        order by datadate desc
        limit {n}
    ) daily_trade;""").iloc[0, 0]


def is_satisfying_liquidity_constraint(required, adv, threshold, in_optimization):
    """Return difference or return true if it satisfies trading liquidity constraints."""

    liquidity_ratio = np.abs(required) / adv
    difference = threshold - liquidity_ratio

    if in_optimization:
        return difference
    else:
        return np.all(difference >= 0.0)


def get_daily_price(api, gvkey, iid, adjust_dividend, start=None, end=None):
    """Return daily price with dividend adjusted or not."""

    if start is not None:
        start_condition = f" and sec_dprc.datadate >= '{start}'"
    else:
        start_condition = ""

    if end is not None:
        end_condition = f" and sec_dprc.datadate <= '{end}'"
    else:
        end_condition = ""

    if not adjust_dividend:
        return api.compustat.read_sql(f"""
        select sec_dprc.datadate, prccd / nullif(coalesce(qunit, 1) * coalesce(ajexdi, 1), 0) as price
        from sec_dprc
        inner join cssecurity cs
        on
            sec_dprc.gvkey = cs.gvkey
            and sec_dprc.iid = cs.iid
            and sec_dprc.datadate between cs.effdate and cs.thrudate
        where 
            sec_dprc.gvkey = '{gvkey}'
            and sec_dprc.iid = '{iid}'
            and cs.secstat = 'A'
            and cs.exchg in ('11', '14')
            and cs.excntry = 'USA'
            {start_condition}
            {end_condition}
        order by sec_dprc.datadate
        ;""").set_index("datadate")

    else:
        return api.compustat.read_sql(f"""
        select sec_dprc.datadate, prccd / nullif(coalesce(qunit, 1) * coalesce(ajexdi, 1), 0) * coalesce(trfd, 1) as price
        from sec_dprc
        left join sec_dtrt
        on 
            sec_dprc.gvkey = sec_dtrt.gvkey 
            and sec_dprc.iid = sec_dtrt.iid 
            and sec_dprc.datadate between sec_dtrt.datadate and coalesce(sec_dtrt.thrudate, '9999-12-31')
        inner join cssecurity cs
        on
            sec_dprc.gvkey = cs.gvkey
            and sec_dprc.iid = cs.iid
            and sec_dprc.datadate between cs.effdate and cs.thrudate
        where 
            sec_dprc.gvkey = '{gvkey}'
            and sec_dprc.iid = '{iid}'
            and cs.secstat = 'A'
            and cs.exchg in ('11', '14')
            and cs.excntry = 'USA'
            {start_condition}
            {end_condition}
        order by sec_dprc.datadate
        ;""").set_index("datadate")


def get_operating_days(market="XNYS"):
    """Return Trading date list."""
    market_calendar = trading_calendars.get_calendar(market)
    holidays = market_calendar.regular_holidays.holidays().append(
        pd.Index(market_calendar.adhoc_holidays).tz_localize(None)
    )
    next_30days = pd.Timestamp(pd.Timestamp.today().date()) + pd.Timedelta(days=30)
    return pd.date_range("1990-01-01", next_30days, freq="B").difference(holidays)


OPERATING_DAYS = get_operating_days()


def change_date_to_the_latest_operating_date(date):
    global OPERATING_DAYS
    return OPERATING_DAYS[OPERATING_DAYS <= date][-1]


def change_date_to_the_lagged_operating_date(date, n):
    global OPERATING_DAYS

    if not (isinstance(n, int) and n >= 0):
        raise ValueError("n should be natural number.")

    latest_operating_date = change_date_to_the_latest_operating_date(date)
    index = OPERATING_DAYS.get_loc(latest_operating_date)
    return OPERATING_DAYS[index + n]


def get_daily_portfolio_weight_and_return_from_month_end_weight(monthly_portfolio_weight, daily_return, n, return_turnover=False):
    """Return daily weight and daily return from monthly weight."""
    monthly_portfolio_weight = monthly_portfolio_weight.copy()
    monthly_portfolio_weight.index = monthly_portfolio_weight.index.map(
        lambda e: change_date_to_the_lagged_operating_date(e, n)
    )
    weight_existence_date = monthly_portfolio_weight.index

    if not daily_return.iloc[0].isna().all():
        raise ValueError("Please do not delete the first price date.")

    if not monthly_portfolio_weight.index.isin(daily_return.index).all():
        import warnings
        print(monthly_portfolio_weight.index[~monthly_portfolio_weight.index.isin(daily_return.index)])
        warnings.warn("Some weight date is not contained in daily return date.")

    first_weight_date = monthly_portfolio_weight.index[0]
    daily_return = daily_return.loc[daily_return.index >= first_weight_date]
    daily_portfolio_weight = monthly_portfolio_weight.reindex(daily_return.index)
    daily_portfolio_return = dict()

    turnover_series = {}
    index = daily_portfolio_weight.index
    for previous_date, current_date in zip(index[:-1], index[1:]):
        previous_weight = daily_portfolio_weight.loc[previous_date]
        current_returns = daily_return.loc[current_date]

        non_standardized_as_is_weight = previous_weight * (1 + current_returns)
        standardized_as_is_weight = non_standardized_as_is_weight / non_standardized_as_is_weight.sum()
        daily_portfolio_return[current_date] = non_standardized_as_is_weight.sum() - 1.0

        if current_date not in weight_existence_date:
            daily_portfolio_weight.loc[current_date] = standardized_as_is_weight
        else:
            turnover_series[current_date] = daily_portfolio_weight.loc[current_date].fillna(0.0) - standardized_as_is_weight.fillna(0.0)

    daily_portfolio_return = pd.Series(daily_portfolio_return, name="daily_return").sort_index()

    if not return_turnover:
        return daily_portfolio_weight, daily_portfolio_return
    else:
        turnover_series_of_stocks = pd.DataFrame.from_dict(turnover_series, orient="index").sort_index()
        return daily_portfolio_weight, daily_portfolio_return, turnover_series_of_stocks


def retrieve_current_ticker(api, gvkey, iid):
    """Return current ticker."""

    return api.compustat.read_sql(f"select tic from security where gvkey='{gvkey}' and iid='{iid}';").iloc[0, 0]


def retrieve_current_company(api, gvkey):
    """Return current company name."""

    return api.compustat.read_sql(f"select conm from company where gvkey='{gvkey}';").iloc[0, 0]


def retrieve_current_sector(api, gvkey):
    """Return current sector."""

    return api.compustat.read_sql(f"""
    select gicdesc 
    from r_giccd 
    where giccd = (
        select gsector 
        from company
        where gvkey='{gvkey}'
        )
    ;""").iloc[0, 0]


def retrieve_current_industry(api, gvkey):
    """Return current sector."""

    return api.compustat.read_sql(f"""
    select gicdesc 
    from r_giccd 
    where giccd = (
        select gind 
        from company
        where gvkey='{gvkey}'
        )
    ;""").iloc[0, 0]


def get_map_from_gvkey_iid_to_datastream_ticker(api, gvkey_iid_list):
    """Return map between gvkey_iid and datastream ticker."""

    class EXCHANGE(Enum):
        NYSE = "LNYSEALL"
        NASDAQ = "LNASCOMP"

    def get_listed_stocks(exchange):
        """Return Datastream Ticker and Exchange Ticker."""
        nonlocal api

        if exchange not in EXCHANGE:
            raise ValueError("exchange should be EXCHANGE item.")

        stocks = api.datastream.get_static_data(exchange.value, ["MNEM", "WC05601"], is_constituents=True)
        stocks = stocks.drop("Date", axis=1)

        columns_needed = ["Instrument", "Value"]
        datastream_ticker_column = "datastream_ticker"
        exchange_ticker_column = "exchange_ticker"

        datastream_tickers = (
            stocks.loc[stocks["DataType"] == "MNEM"][columns_needed]
            .rename({"Value": datastream_ticker_column}, axis=1)
        )
        exchange_tickers = (
            stocks.loc[stocks["DataType"] == "WC05601"][columns_needed]
            .rename({"Value": exchange_ticker_column}, axis=1)
        )
        tickers = pd.merge(exchange_tickers, datastream_tickers, "outer", "Instrument", validate="one_to_one")

        if tickers[datastream_ticker_column].isnull().any():
            invalid_case_filter = tickers[datastream_ticker_column].isnull()
            invalid_cases = tickers.loc[invalid_case_filter]
            tickers = tickers.loc[~invalid_case_filter]
            print(f"datastream ticker not exist: {len(invalid_cases)}cases")
            print(invalid_cases)

        if tickers[exchange_ticker_column].isnull().any():
            invalid_case_filter = tickers[exchange_ticker_column].isnull()
            invalid_cases = tickers.loc[invalid_case_filter]
            tickers = tickers.loc[~invalid_case_filter]
            print(f"exchange ticker not exist: {len(invalid_cases)}cases")
            print(invalid_cases)

        if tickers[datastream_ticker_column].duplicated().sum():
            duplicated_case_filter = tickers["datastream_ticker"].duplicated(keep=False)
            duplicated_cases = tickers.loc[duplicated_case_filter]
            print(f"datastream ticker is duplicated: {len(duplicated_cases)}cases")
            print(duplicated_cases)

        if tickers[exchange_ticker_column].duplicated().sum():
            duplicated_case_filter = tickers[exchange_ticker_column].duplicated(keep=False)
            duplicated_cases = tickers.loc[duplicated_case_filter]
            print(f"exchange ticker is duplicated: {len(duplicated_cases)}cases")
            print(duplicated_cases)

        tickers = tickers.dropna(axis=0, how="any").set_index("exchange_ticker").drop("Instrument", axis=1).iloc[:, 0]

        return tickers

    nyse = get_listed_stocks(EXCHANGE.NYSE)
    nasdaq = get_listed_stocks(EXCHANGE.NASDAQ)
    us_stocks = pd.concat([nyse, nasdaq])
    us_stocks_stripped = us_stocks.map(lambda e: e.replace("U:", "").replace("@", ""))

    tickers_mapping = {}
    for gvkey_iid in gvkey_iid_list:
        ticker = api.compustat.get_current_meta(gvkey_iid).loc['tic'][:-len(" US EQUITY")]

        try:
            founded_value = us_stocks.loc[ticker]

        except KeyError as e:
            print(f"Cannot find. Instead search in stripped on {gvkey_iid}({ticker}).")

            proxy_matched = us_stocks_stripped == ticker
            if proxy_matched.sum() == 1:
                tickers_mapping[gvkey_iid] = us_stocks.loc[proxy_matched].item()

            elif proxy_matched.sum() == 0:
                raise e

            elif proxy_matched.sum() > 1:
                proxy_matched_values = us_stocks.loc[proxy_matched]
                for i, v in enumerate(proxy_matched_values):
                    print(f"{i}: {v}")
                print()

                while True:
                    answer = input("Type exact ticker name:")
                    selected = proxy_matched_values == answer
                    if selected.sum() == 0:
                        print(f"No match found: {answer}")
                    elif selected.sum() > 1:
                        print(f"Duplicated match found: {answer}")
                    else:
                        tickers_mapping[gvkey_iid] = answer
                        break

        else:
            if isinstance(founded_value, str):
                tickers_mapping[gvkey_iid] = founded_value

            elif isinstance(founded_value, pd.Series):
                print(f"Duplicated Cases occurred on {gvkey_iid}({ticker}).")
                print("Choose one item in below.")

                for i, v in enumerate(founded_value):
                    print(f"{i}: {v}")
                print()

                while True:
                    answer = input("Type exact ticker name:")
                    matched = founded_value == answer

                    if matched.sum() == 0:
                        print(f"No match found: {answer}")
                    elif matched.sum() > 1:
                        print(f"Duplicated match found: {answer}")
                    else:
                        tickers_mapping[gvkey_iid] = answer
                        break
            else:
                raise Exception("Unexpected")

    return tickers_mapping


def get_close_prices(api, target_date, datastream_tickers):
    one_year_ago = pd.Timestamp(pd.Timestamp.now().date()) - pd.Timedelta(days=260)

    close_prices = []
    for i in range(0, len(datastream_tickers), 10):
        tickers = datastream_tickers[i:i+10]
        print(i, tickers)
        close_prices.append(
            api.datastream.get_time_series_data(tickers, "P", frequency="D", date_from=one_year_ago).loc[target_date]
        )

    return pd.concat(close_prices)


def get_average_daily_volume(api, gvkey_iid, target_date, n):
    """Return average daily volume($) over n trading days."""
    gvkey, iid = gvkey_iid.split("_")

    if isinstance(target_date, (datetime.datetime, datetime.date)):
        target_date = target_date.strftime("%Y-%m-%d")

    elif isinstance(target_date, str):
        try:
            datetime.datetime.strptime(target_date, '%Y-%m-%d')
        except Exception as e:
            raise e

    else:
        raise TypeError("target_date must be str['%Y-%m-%d'] or datetime.date or datetime.datetime")

    try:
        daily_volumes = api.compustat.read_sql(
                f"""select datadate, prccd * cshtrd as volume 
        from sec_dprc 
        where gvkey='{gvkey}' and iid='{iid}' and datadate <= '{target_date}'
        order by datadate desc
        limit {n}
        ;"""
            )
    except pd.errors.EmptyDataError:
        return None

    daily_volumes = daily_volumes.set_index("datadate")["volume"].sort_index()

    if daily_volumes.index[-1].strftime('%Y-%m-%d') != target_date:
        if gvkey_iid == "001690_01":
            import pdb; pdb.set_trace()
        warnings.warn(f"{gvkey_iid} is not traded on {target_date}.")

    if len(daily_volumes.index) < n:
        warnings.warn(f"{gvkey_iid} is not traded for {n} days.")

    if daily_volumes.index[0] < pd.Timestamp(target_date) - pd.Timedelta(days=n + 4 + int(3/5*n)):
        warnings.warn(f"{gvkey_iid} is infrequently traded. (first date: {daily_volumes.index[0]}, target date: {target_date})")
        return None

    return daily_volumes.mean()
