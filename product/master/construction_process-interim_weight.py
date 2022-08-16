"""Adjusting weight for interim update, not at the first day of month."""
import sys; sys.path.append(".")
from pathlib import Path
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from paths import DATA_DIR


IDENTIFIER = "master-timecheck"
BASKET1_WEIGHT_FILENAME = "latest_basket1.csv"
BASKET2_WEIGHT_FILENAME = "latest_basket2.csv"
START_DATE = "2021-11-30"
PORTFOLIO_DATE = "2021-11-30"
END_DATE = "2021-12-01"


if __name__ == "__main__":
    base_path = Path(DATA_DIR) / IDENTIFIER

    api = Kirin()

    # BASKET1
    basket1_weight = pd.read_csv(base_path / BASKET1_WEIGHT_FILENAME, index_col=0).transpose()
    basket1_weight.index = [pd.Timestamp(PORTFOLIO_DATE)]
    basket1_gvkey_iid = [e.split("_") for e in basket1_weight.columns]
    basket1_dividend_adjusted_price = [
        master_utils.get_daily_price(
            api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=END_DATE
        ) for gvkey, iid in basket1_gvkey_iid
    ]
    basket1_dividend_adjusted_price = pd.concat(basket1_dividend_adjusted_price, axis=1).ffill()
    basket1_dividend_adjusted_price.columns = basket1_weight.columns
    basket1_dividend_adjusted_return = basket1_dividend_adjusted_price.pct_change(fill_method=None)

    daily_basket1_weight, daily_basket1_return = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket1_weight, basket1_dividend_adjusted_return
    )
    daily_basket1_weight.iloc[-1].to_csv(base_path / ("interim_" + BASKET1_WEIGHT_FILENAME))
    daily_basket1_return.to_csv(base_path / ("daily_return_" + BASKET1_WEIGHT_FILENAME))

    # BASKET2
    basket2_weight = pd.read_csv(base_path / BASKET2_WEIGHT_FILENAME, index_col=0).transpose()
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
        basket2_weight, basket2_dividend_adjusted_return
    )
    daily_basket2_weight.iloc[-1].to_csv(base_path / ("interim_" + BASKET2_WEIGHT_FILENAME))
    daily_basket2_return.to_csv(base_path / ("daily_return_" + BASKET2_WEIGHT_FILENAME))
