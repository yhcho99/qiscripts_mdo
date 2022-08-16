"""Daily AI performance."""
import sys; sys.path.append(".")
from pathlib import Path
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from paths import DATA_DIR


# IDENTIFIER = "masters/ai99"
BASKET1_WEIGHT_FILENAME = "basket1-pure-weight.csv"
BASKET2_WEIGHT_FILENAME = "basket2-pure-weight.csv"
START_DATE = "2015-12-15"


def to_gvkey_iid(api, tic):
    data = api.compustat.read_sql(f"select gvkey, iid from security where tic='{tic}';").iloc[0]
    gvkey, iid = data.loc["gvkey"], data.loc["iid"]
    return gvkey, iid


if __name__ == "__main__":
    base_path = Path(r"C:\Users\marketing\Desktop\마케팅\master project\phase2\7. Tech Material\Q&A 19 Jan\Weight&Prices")

    api = Kirin()

    # BASKET1
    basket1_weight = pd.read_csv(base_path / BASKET1_WEIGHT_FILENAME, index_col=0, parse_dates=True)
    basket1_gvkey_iid = [to_gvkey_iid(api, e) for e in basket1_weight.columns]
    # basket1_gvkey_iid = [e.split("_") for e in basket1_weight.columns]
    basket1_dividend_adjusted_price = [
        master_utils.get_daily_price(
            api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None
        ) for gvkey, iid in basket1_gvkey_iid
    ]
    basket1_dividend_adjusted_price = pd.concat(basket1_dividend_adjusted_price, axis=1).ffill()
    basket1_dividend_adjusted_price.columns = basket1_weight.columns
    basket1_dividend_adjusted_return = basket1_dividend_adjusted_price.pct_change(fill_method=None)

    daily_basket1_weight, daily_basket1_return = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket1_weight, basket1_dividend_adjusted_return, 1
    )
    basket1_dividend_adjusted_price.to_csv(base_path / "basket1_prices.csv")
    daily_basket1_weight.to_csv(base_path / ("daily_weight_" + BASKET1_WEIGHT_FILENAME))
    daily_basket1_return.to_csv(base_path / ("daily_return_" + BASKET1_WEIGHT_FILENAME))

    # BASKET2
    basket2_weight = pd.read_csv(base_path / BASKET2_WEIGHT_FILENAME, index_col=0, parse_dates=True)
    basket2_gvkey_iid = [to_gvkey_iid(api, e) for e in basket2_weight.columns]
    # basket2_gvkey_iid = [e.split("_") for e in basket2_weight.columns]
    basket2_dividend_adjusted_price = [
        master_utils.get_daily_price(
            api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None
        ) for gvkey, iid in basket2_gvkey_iid
    ]
    basket2_dividend_adjusted_price = pd.concat(basket2_dividend_adjusted_price, axis=1).ffill()
    basket2_dividend_adjusted_price.columns = basket2_weight.columns
    basket2_dividend_adjusted_return = basket2_dividend_adjusted_price.pct_change(fill_method=None)

    daily_basket2_weight, daily_basket2_return = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
        basket2_weight, basket2_dividend_adjusted_return, 1
    )
    basket2_dividend_adjusted_price.to_csv(base_path / "basket2_prices.csv")
    daily_basket2_weight.to_csv(base_path / ("daily_weight_" + BASKET2_WEIGHT_FILENAME))
    daily_basket2_return.to_csv(base_path / ("daily_return_" + BASKET2_WEIGHT_FILENAME))
