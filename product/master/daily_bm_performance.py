"""Daily BM performance."""
import sys; sys.path.append(".")
from pathlib import Path
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from paths import DATA_DIR
from product.master.baskets import *


START_DATE = "2015-12-15"


if __name__ == "__main__":
    base_path = Path(r"C:\Users\marketing\Desktop\마케팅\master project\phase2\7. Tech Material\Q&A 19 Jan\Weight&Prices")
    base_path.mkdir(parents=True, exist_ok=True)

    api = Kirin()
    meta = api.compustat.set_investment_universe(
        exchange=["NYSE", "NASDAQ", "OTC"],
        security_type=["COMMON", "ADR"],
        class_A_only=False, primary_issue=False
    )
    mv = api.compustat.get_monthly_market_value()

    # BASKET1
    basket1_gvkey_iid = [e.split("_") for e in BASKET1]
    basket1_dividend_adjusted_price = [
        master_utils.get_daily_price(
            api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None
        ) for gvkey, iid in basket1_gvkey_iid
    ]
    basket1_dividend_adjusted_price = pd.concat(basket1_dividend_adjusted_price, axis=1).ffill()
    basket1_dividend_adjusted_price.columns = BASKET1
    basket1_dividend_adjusted_return = basket1_dividend_adjusted_price.pct_change(fill_method=None)

    basket1_mv = mv.loc[START_DATE:, BASKET1]
    basket1_weight = basket1_mv.div(basket1_mv.sum(axis=1), axis=0)

    daily_basket1_weight, daily_basket1_return = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket1_weight, basket1_dividend_adjusted_return, 1
    )
    basket1_weight.to_csv(base_path / "monthly_weight_basket1.csv")
    daily_basket1_weight.to_csv(base_path / "weight_basket1.csv")
    daily_basket1_return.to_csv(base_path / "return_basket1.csv")

    # BASKET2
    basket2_gvkey_iid = [e.split("_") for e in BASKET2]
    basket2_dividend_adjusted_price = [
        master_utils.get_daily_price(
            api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None
        ) for gvkey, iid in basket2_gvkey_iid
    ]
    basket2_dividend_adjusted_price = pd.concat(basket2_dividend_adjusted_price, axis=1).ffill()
    basket2_dividend_adjusted_price.columns = BASKET2
    basket2_dividend_adjusted_return = basket2_dividend_adjusted_price.pct_change(fill_method=None)

    basket2_mv = mv.loc[START_DATE:, BASKET2]
    basket2_weight = basket2_mv.div(basket2_mv.sum(axis=1), axis=0)

    daily_basket2_weight, daily_basket2_return = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket2_weight, basket2_dividend_adjusted_return, 1
    )
    basket2_weight.to_csv(base_path / "monthly_weight_basket2.csv")
    daily_basket2_weight.to_csv(base_path / "weight_basket2.csv")
    daily_basket2_return.to_csv(base_path / "return_basket2.csv")
