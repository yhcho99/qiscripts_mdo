"""Formatting weight for GS."""
import sys; sys.path.append(".")
from pathlib import Path
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from product.master.baskets import BASKET1, BASKET2
from paths import DATA_DIR


DATE = "2021-11-30"
IDENTIFIER = f"assets/masters/prices/{DATE}"


if __name__ == "__main__":
    base_path = Path(DATA_DIR) / IDENTIFIER
    base_path.mkdir(parents=True, exist_ok=True)

    api = Kirin()
    all_stocks_in_basket = list(set(BASKET1).union(set(BASKET2)))

    map_gvkey_iid_to_datastream_ticker = master_utils.get_map_from_gvkey_iid_to_datastream_ticker(
        api, all_stocks_in_basket
    )
    map_gvkey_iid_to_bbg_ticker = {
        gvkey_iid: master_utils.get_latest_bbg_ticker(api, *gvkey_iid.split("_"))
        for gvkey_iid in all_stocks_in_basket
    }

    close_prices = master_utils.get_close_prices(api, DATE, list(map_gvkey_iid_to_datastream_ticker.values()))

    basket1_close_prices = {}
    for gvkey_iid in BASKET1:
        datastream_ticker = map_gvkey_iid_to_datastream_ticker[gvkey_iid]
        bbg_ticker = map_gvkey_iid_to_bbg_ticker[gvkey_iid]
        basket1_close_prices[bbg_ticker] = close_prices.loc[datastream_ticker]
    basket1_close_prices = pd.Series(basket1_close_prices, name="Close")
    basket1_close_prices.to_csv(base_path / "BASKET1_CLOSE.csv")

    basket2_close_prices = {}
    for gvkey_iid in BASKET2:
        datastream_ticker = map_gvkey_iid_to_datastream_ticker[gvkey_iid]
        bbg_ticker = map_gvkey_iid_to_bbg_ticker[gvkey_iid]
        basket2_close_prices[bbg_ticker] = close_prices.loc[datastream_ticker]
    basket2_close_prices = pd.Series(basket2_close_prices, name="Close")
    basket2_close_prices.to_csv(base_path / "BASKET2_CLOSE.csv")
