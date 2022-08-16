"""Daily BM performance."""
import sys; sys.path.append(".")
from pathlib import Path
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from product.master.baskets import *


START_DATE = "2021-12-15"


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

    # BASKET1
    basket1_gvkey_iid = [e.split("_") for e in basket1_weight.columns]
    basket1_dividend_adjusted_price = [
        master_utils.get_daily_price(
            api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None
        ) for gvkey, iid in basket1_gvkey_iid
    ]
    basket1_dividend_adjusted_price = pd.concat(basket1_dividend_adjusted_price, axis=1).ffill()
    basket1_dividend_adjusted_price.columns = basket1_weight.columns
    basket1_dividend_adjusted_return = basket1_dividend_adjusted_price.pct_change(fill_method=None)

    daily_basket1_weight, daily_basket1_return = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket1_weight, basket1_dividend_adjusted_return, 0
    )
    current_date = daily_basket1_weight.index[-1].strftime('%Y-%m-%d')
    current_basket1_weight = daily_basket1_weight.iloc[-1].dropna()
    current_basket1_weight.index = current_basket1_weight.index.map(lambda e: master_utils.get_latest_bbg_ticker(api, *e.split("_")))
    current_basket1_weight.to_csv(base_path / f"basket1-weight({current_date}).csv")

    # BASKET2
    basket2_weight = pd.read_csv(base_path / "Rebalancing History Basket2.csv", index_col=0, parse_dates=True).loc[START_DATE:].dropna(how="all", axis=1)
    basket2_weight = basket2_weight.rename(MAPPING, axis=1)

    basket2_gvkey_iid = [e.split("_") for e in basket2_weight.columns]
    basket2_dividend_adjusted_price = [
        master_utils.get_daily_price(
            api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None
        ) for gvkey, iid in basket2_gvkey_iid
    ]
    basket2_dividend_adjusted_price = pd.concat(basket2_dividend_adjusted_price, axis=1).ffill()
    basket2_dividend_adjusted_price.columns = basket2_weight.columns
    basket2_dividend_adjusted_return = basket2_dividend_adjusted_price.pct_change(fill_method=None)

    daily_basket2_weight, daily_basket2_return = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket2_weight, basket2_dividend_adjusted_return, 0
    )
    current_date = daily_basket2_weight.index[-1].strftime('%Y-%m-%d')
    current_basket2_weight = daily_basket2_weight.iloc[-1].dropna()
    current_basket2_weight.index = current_basket2_weight.index.map(lambda e: master_utils.get_latest_bbg_ticker(api, *e.split("_")))
    current_basket2_weight.to_csv(base_path / f"basket2-weight({current_date}).csv")
