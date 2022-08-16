"""Daily BM performance."""
import sys; sys.path.append(".")
from pathlib import Path
import operator
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from product.master.baskets import *


START_DATE = "2020-12-15"
ASSUMED_START_AUM = 100_000_000
ASSUMED_AUM_LIST = None  # [100_000_000] * 12
REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}


if __name__ == "__main__":
    base_path = Path(r"C:\Users\marketing\Desktop\마케팅\master project\phase2\Master Ops\weight_220201")

    api = Kirin()
    meta = api.compustat.set_investment_universe(
        exchange=["NYSE", "NASDAQ", "OTC"],
        security_type=["COMMON", "ADR"],
        class_A_only=False, primary_issue=False
    )

    basket1_weight = pd.read_csv(base_path / "Rebalancing History Basket1.csv", index_col=0, parse_dates=True).loc[START_DATE:].dropna(how="all", axis=1)
    basket1_weight = basket1_weight.rename(MAPPING, axis=1)

    new_index = []
    for date in basket1_weight.index:
        latest_operating_date = master_utils.change_date_to_the_latest_operating_date(date)
        if date.strftime('%Y-%m-%d') == latest_operating_date.strftime('%Y-%m-%d'):
            new_index.append(latest_operating_date)
        else:
            new_index.append(master_utils.change_date_to_the_lagged_operating_date(date, 1))
    basket1_weight.index = new_index
    print(basket1_weight)

    # BASKET1
    basket1_gvkey_iid = [e.split("_") for e in basket1_weight.columns]
    basket1_dividend_adjusted_price = [
        master_utils.get_daily_price(api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None)
        for gvkey, iid in basket1_gvkey_iid
    ]
    basket1_dividend_adjusted_price = pd.concat(basket1_dividend_adjusted_price, axis=1).ffill()
    basket1_dividend_adjusted_price.columns = basket1_weight.columns
    basket1_dividend_adjusted_return = basket1_dividend_adjusted_price.pct_change(fill_method=None)

    _, daily_basket1_return, basket1_turnover_stocks = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket1_weight, basket1_dividend_adjusted_return, 0, return_turnover=True
    )
    basket1_turnover_stocks = basket1_turnover_stocks.rename(REVERSE_MAPPING, axis=1)
    basket1_positive_orders = basket1_turnover_stocks[basket1_turnover_stocks > 0.0].sum(axis=1)
    basket1_negative_orders = basket1_turnover_stocks[basket1_turnover_stocks < 0.0].sum(axis=1)
    basket1_total_orders = basket1_positive_orders + basket1_negative_orders.abs()
    basket1_turnovers = pd.concat([basket1_total_orders, basket1_positive_orders, basket1_negative_orders, basket1_turnover_stocks], axis=1)
    basket1_turnovers.columns = ["Total Transactions", "Positive Transactions", "Negative Transactions"] + basket1_turnover_stocks.columns.to_list()
    basket1_turnovers.to_csv(base_path / "basket1-turnover(percentage).csv")

    basket1_notional = (daily_basket1_return + 1.0).cumprod().reindex(basket1_turnover_stocks.index)
    if not operator.xor(ASSUMED_START_AUM is None, ASSUMED_AUM_LIST is None):
        raise ValueError("One of them should be None")

    if ASSUMED_AUM_LIST is not None:
        index_values = basket1_notional.values
        if len(index_values) != len(ASSUMED_AUM_LIST):
            raise ValueError
        basket1_notional.loc[:] = [i_value * a_aum for i_value, a_aum in zip(index_values, ASSUMED_AUM_LIST)]

    if ASSUMED_START_AUM is not None:
        basket1_notional *= ASSUMED_START_AUM

    basket1_turnovers_notional = basket1_turnovers.multiply(basket1_notional, axis=0)
    basket1_turnovers_notional_w_aum = pd.concat([basket1_notional, basket1_turnovers_notional], axis=1)
    basket1_turnovers_notional_w_aum.columns = ["AUM"] + basket1_turnovers_notional.columns.to_list()
    basket1_turnovers_notional_w_aum.to_csv(base_path / "basket1-turnover(notional).csv")

    # BASKET2
    basket2_weight = pd.read_csv(base_path / "Rebalancing History Basket2.csv", index_col=0, parse_dates=True).loc[START_DATE:].dropna(how="all", axis=1)
    basket2_weight = basket2_weight.rename(MAPPING, axis=1)

    new_index = []
    for date in basket2_weight.index:
        latest_operating_date = master_utils.change_date_to_the_latest_operating_date(date)
        if date == latest_operating_date:
            new_index.append(latest_operating_date)
        else:
            new_index.append(master_utils.change_date_to_the_lagged_operating_date(date, 1))
    basket2_weight.index = new_index

    basket2_gvkey_iid = [e.split("_") for e in basket2_weight.columns]
    basket2_dividend_adjusted_price = [
        master_utils.get_daily_price(api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None)
        for gvkey, iid in basket2_gvkey_iid
    ]
    basket2_dividend_adjusted_price = pd.concat(basket2_dividend_adjusted_price, axis=1).ffill()
    basket2_dividend_adjusted_price.columns = basket2_weight.columns
    basket2_dividend_adjusted_return = basket2_dividend_adjusted_price.pct_change(fill_method=None)

    _, daily_basket2_return, basket2_turnover_stocks = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket2_weight, basket2_dividend_adjusted_return, 0, return_turnover=True
    )
    basket2_turnover_stocks = basket2_turnover_stocks.rename(REVERSE_MAPPING, axis=1)
    basket2_positive_orders = basket2_turnover_stocks[basket2_turnover_stocks > 0.0].sum(axis=1)
    basket2_negative_orders = basket2_turnover_stocks[basket2_turnover_stocks < 0.0].sum(axis=1)
    basket2_total_orders = basket2_positive_orders + basket2_negative_orders.abs()
    basket2_turnovers = pd.concat([basket2_total_orders, basket2_positive_orders, basket2_negative_orders, basket2_turnover_stocks], axis=1)
    basket2_turnovers.columns = ["Total Transactions", "Positive Transactions", "Negative Transactions"] + basket2_turnover_stocks.columns.to_list()
    basket2_turnovers.to_csv(base_path / "basket2-turnover(percentage).csv")

    basket2_notional = (daily_basket2_return + 1.0).cumprod().reindex(basket2_turnover_stocks.index)
    if not operator.xor(ASSUMED_START_AUM is None, ASSUMED_AUM_LIST is None):
        raise ValueError("One of them should be None")

    if ASSUMED_AUM_LIST is not None:
        index_values = basket2_notional.values
        if len(index_values) != len(ASSUMED_AUM_LIST):
            raise ValueError
        basket2_notional.loc[:] = [i_value * a_aum for i_value, a_aum in zip(index_values, ASSUMED_AUM_LIST)]

    if ASSUMED_START_AUM is not None:
        basket2_notional *= ASSUMED_START_AUM

    basket2_turnovers_notional = basket2_turnovers.multiply(basket2_notional, axis=0)
    basket2_turnovers_notional_w_aum = pd.concat([basket2_notional, basket2_turnovers_notional], axis=1)
    basket2_turnovers_notional_w_aum.columns = ["AUM"] + basket2_turnovers_notional.columns.to_list()
    basket2_turnovers_notional_w_aum.to_csv(base_path / "basket2-turnover(notional).csv")
