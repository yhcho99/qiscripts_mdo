"""Daily BM performance."""
import sys; sys.path.append(".")
from pathlib import Path
import numpy as np
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from product.master.baskets import *


ADV_N = 20
START_DATE = "2021-12-15"
THRESHOLD = 0.5
ALERT_MSG_MAPPING = {
    True: "ADV Drop Too much",
    False: "Normal"
}
REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}


if __name__ == "__main__":
    base_path = Path(r"C:\Users\marketing\Desktop\마케팅\master project\phase2\Master Ops\weight_220201")

    api = Kirin()
    meta = api.compustat.set_investment_universe(
        exchange=["NYSE", "NASDAQ", "OTC"],
        security_type=["COMMON", "ADR"],
        class_A_only=False, primary_issue=False
    )

    # BASKET1
    basket1_weight = pd.read_csv(base_path / "Rebalancing History Basket1.csv", index_col=0, parse_dates=True).loc[START_DATE:]
    basket1_weight = basket1_weight.rename(MAPPING, axis=1)

    new_index = []
    for date in basket1_weight.index:
        latest_operating_date = master_utils.change_date_to_the_latest_operating_date(date)
        if date == latest_operating_date:
            new_index.append(latest_operating_date)
        else:
            new_index.append(master_utils.change_date_to_the_lagged_operating_date(date, 1))
    basket1_weight.index = new_index

    basket1_adv_history = {}
    for date in basket1_weight.index:
        date_history = {}
        for gvkey_iid in basket1_weight.columns:
            w = basket1_weight.loc[date, gvkey_iid]
            if w is not None and np.isfinite(w):
                date_history[gvkey_iid] = master_utils.get_average_daily_volume(api, gvkey_iid, date, ADV_N)

        basket1_adv_history[date] = date_history
    basket1_adv_history = pd.DataFrame.from_dict(basket1_adv_history).sort_index(axis=1, ascending=False)
    basket1_adv_history.index = basket1_adv_history.index.map(lambda e: master_utils.get_latest_bbg_ticker(api, *e.split("_")))
    basket1_ratio = basket1_adv_history.iloc[:, 0] / basket1_adv_history.iloc[:, 1]
    basket1_alert = (basket1_ratio < THRESHOLD).apply(lambda e: ALERT_MSG_MAPPING[e])
    basket1_alert_frame = pd.concat([basket1_alert, basket1_ratio, basket1_adv_history], axis=1)
    basket1_alert_frame.columns = ["Alert Message", "Current ADV/Previous ADV"] + basket1_adv_history.columns.to_list()
    basket1_alert_frame = basket1_alert_frame.sort_values("Current ADV/Previous ADV")
    basket1_alert_frame.to_csv(base_path / "basket1-adv-check.csv")

    # BASKET2
    basket2_weight = pd.read_csv(base_path / "Rebalancing History Basket2.csv", index_col=0, parse_dates=True).loc[START_DATE:]
    basket2_weight = basket2_weight.rename(MAPPING, axis=1)

    new_index = []
    for date in basket2_weight.index:
        latest_operating_date = master_utils.change_date_to_the_latest_operating_date(date)
        if date == latest_operating_date:
            new_index.append(latest_operating_date)
        else:
            new_index.append(master_utils.change_date_to_the_lagged_operating_date(date, 1))
    basket2_weight.index = new_index

    basket2_adv_history = {}
    for date in basket2_weight.index:
        date_history = {}
        for gvkey_iid in basket2_weight.columns:
            w = basket2_weight.loc[date, gvkey_iid]
            if w is not None and np.isfinite(w):
                date_history[gvkey_iid] = master_utils.get_average_daily_volume(api, gvkey_iid, date, ADV_N)

        basket2_adv_history[date] = date_history
    basket2_adv_history = pd.DataFrame.from_dict(basket2_adv_history).sort_index(axis=1, ascending=False)
    basket2_adv_history.index = basket2_adv_history.index.map(lambda e: master_utils.get_latest_bbg_ticker(api, *e.split("_")))
    basket2_ratio = basket2_adv_history.iloc[:, 0] / basket2_adv_history.iloc[:, 1]
    basket2_alert = (basket2_ratio < THRESHOLD).apply(lambda e: ALERT_MSG_MAPPING[e])
    basket2_alert_frame = pd.concat([basket2_alert, basket2_ratio, basket2_adv_history], axis=1)
    basket2_alert_frame.columns = ["Alert Message", "Current ADV/Previous ADV"] + basket2_adv_history.columns.to_list()
    basket2_alert_frame = basket2_alert_frame.sort_values("Current ADV/Previous ADV")
    basket2_alert_frame.to_csv(base_path / "basket2-adv-check.csv")
