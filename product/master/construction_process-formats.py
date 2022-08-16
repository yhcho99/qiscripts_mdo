"""Formatting weight for GS."""
import sys; sys.path.append(".")
from pathlib import Path
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from paths import DATA_DIR


IDENTIFIER = "master_220201"
BASKET1_WEIGHT_FILENAME = "latest_basket1.csv"
BASKET2_WEIGHT_FILENAME = "latest_basket2.csv"

if __name__ == "__main__":
    exp_path = Path(DATA_DIR) / IDENTIFIER

    basket1_path = exp_path / BASKET1_WEIGHT_FILENAME
    basket2_path = exp_path / BASKET2_WEIGHT_FILENAME

    latest_basket1 = pd.read_csv(basket1_path, index_col=0).iloc[0]
    latest_basket2 = pd.read_csv(basket2_path, index_col=0).iloc[0]

    api = Kirin(use_sub_server_for_compustat=True)
    union_index = latest_basket1.index.union(latest_basket2.index)
    union_bbg_ticker = (
        union_index
        .map(lambda e: master_utils.get_latest_bbg_ticker(api, *e.split("_")))
        .map(lambda e: master_utils.post_adjust_bbg_ticker(e))
    )
    union_mapping = pd.Series(union_bbg_ticker, index=union_index, name="BBG Ticker")

    latest_basket1.index = latest_basket1.index.map(lambda e: union_mapping.loc[e])
    latest_basket2.index = latest_basket2.index.map(lambda e: union_mapping.loc[e])

    columns = ["Eligible Component", "Weight"]
    latest_basket1 = latest_basket1.to_frame().reset_index()
    latest_basket1.columns = columns

    latest_basket2 = latest_basket2.to_frame().reset_index()
    latest_basket2.columns = columns

    union_mapping.to_csv(exp_path / "GVKEY_IID to BBG.csv")
    latest_basket1.to_csv(exp_path / "GS-BASKET1.csv", index=False)
    latest_basket2.to_csv(exp_path / "GS-BASKET2.csv", index=False)
