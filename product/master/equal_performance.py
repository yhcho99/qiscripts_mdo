"""Daily Equal performance."""
import sys; sys.path.append(".")
from pathlib import Path
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from product.master.baskets import BASKET1, BASKET2
from paths import DATA_DIR


IDENTIFIER = "masters"
BASKET1_WEIGHT_FILENAME = "equal-basket1.csv"
BASKET2_WEIGHT_FILENAME = "equal-basket2.csv"
START_DATE = "2015-12-15"


if __name__ == "__main__":
    base_path = Path(r"C:\Users\marketing\Desktop\마케팅\master project\phase2\7. Tech Material\Q&A 19 Jan\Weight&Prices\eq2")

    api = Kirin()
    meta = api.compustat.set_investment_universe(
        exchange=["NYSE", "NASDAQ", "OTC"],
        security_type=["COMMON", "ADR"],
        class_A_only=False,
        primary_issue=False
    )
    mv = api.compustat.get_monthly_market_value().loc[START_DATE:]

    basket1_mask = mv.loc[:, BASKET1].notna().astype(float)
    basket1_weight = basket1_mask.div(basket1_mask.sum(1), 0)

    # BASKET1
    basket1_dividend_adjusted_price = [
        master_utils.get_daily_price(
            api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None
        ) for gvkey, iid in map(lambda e: e.split("_"), BASKET1)
    ]
    basket1_dividend_adjusted_price = pd.concat(basket1_dividend_adjusted_price, axis=1).ffill()
    basket1_dividend_adjusted_price.columns = basket1_weight.columns
    basket1_dividend_adjusted_return = basket1_dividend_adjusted_price.pct_change(fill_method=None)

    daily_basket1_weight, daily_basket1_return = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket1_weight, basket1_dividend_adjusted_return, 0
    )
    basket1_weight.to_csv(base_path / "basket1_equal_weight.csv")
    daily_basket1_weight.to_csv(base_path / ("daily_weight_" + BASKET1_WEIGHT_FILENAME))
    daily_basket1_return.to_csv(base_path / ("daily_return_" + BASKET1_WEIGHT_FILENAME))

    # BASKET2
    basket2_mask = mv.loc[:, BASKET2].notna().astype(float)
    basket2_weight = basket2_mask.div(basket2_mask.sum(1), 0)

    basket2_dividend_adjusted_price = [
        master_utils.get_daily_price(
            api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None
        ) for gvkey, iid in map(lambda e: e.split("_"), BASKET2)
    ]
    basket2_dividend_adjusted_price = pd.concat(basket2_dividend_adjusted_price, axis=1).ffill()
    basket2_dividend_adjusted_price.columns = basket2_weight.columns
    basket2_dividend_adjusted_return = basket2_dividend_adjusted_price.pct_change(fill_method=None)

    daily_basket2_weight, daily_basket2_return = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket2_weight, basket2_dividend_adjusted_return, 0
    )
    basket2_weight.to_csv(base_path / "basket2_equal_weight.csv")
    daily_basket2_weight.to_csv(base_path / ("daily_weight_" + BASKET2_WEIGHT_FILENAME))
    daily_basket2_return.to_csv(base_path / ("daily_return_" + BASKET2_WEIGHT_FILENAME))
