"""Make Zero weight BBG Ticker weight file for the initiation."""
import sys; sys.path.append(".")
from pathlib import Path
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from paths import DATA_DIR


IDENTIFIER = "master-220228"


if __name__ == "__main__":
    exp_path = Path(DATA_DIR) / IDENTIFIER

    # 1. Read Target Portfolio
    print("Read Target Portfolio")
    latest_basket1_path = exp_path / "strategy" / "combined.csv"
    latest_basket2_path = exp_path / "intermediate" / "BASKET2" / "strategy" / "combined.csv"

    latest_basket1 = pd.read_csv(latest_basket1_path, index_col=0, parse_dates=True).iloc[-1].dropna()
    latest_basket2 = pd.read_csv(latest_basket2_path, index_col=0, parse_dates=True).iloc[-1].dropna()

    print("Transform GVKEY IID to BBG Ticker")
    api = Kirin(use_sub_server_for_compustat=True)
    latest_basket1.index = latest_basket1.index.map(lambda e: master_utils.get_latest_bbg_ticker(api, *e.split("_")))
    latest_basket2.index = latest_basket2.index.map(lambda e: master_utils.get_latest_bbg_ticker(api, *e.split("_")))

    latest_basket1.loc[:] = 0.0
    latest_basket2.loc[:] = 0.0

    latest_basket1 = latest_basket1.reset_index()
    latest_basket2 = latest_basket2.reset_index()

    columns = ["Eligible Component", "Weight"]
    latest_basket1.columns = columns
    latest_basket2.columns = columns

    previous_basket1_path = exp_path / "previous_basket1.csv"
    previous_basket2_path = exp_path / "previous_basket2.csv"

    latest_basket1.to_csv(previous_basket1_path, index=False)
    latest_basket2.to_csv(previous_basket2_path, index=False)
