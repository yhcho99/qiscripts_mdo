"""Formatting return and weight for Dashboard."""
import sys; sys.path.append(".")
from pathlib import Path
import numpy as np
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from paths import DATA_DIR
from product.master.baskets import *


IDENTIFIER = "assets/masters"
BASKET1_WEIGHT_FILENAME = "basket1-bm-weight.csv"
BASKET1_RETURN_FILENAME = "basket1-bm-return.csv"
BASKET2_WEIGHT_FILENAME = "basket2-bm-weight.csv"
BASKET2_RETURN_FILENAME = "basket2-bm-return.csv"
START_DATE = "2015-12-15"


def transform_weight_for_dashboard_db(weight, strategy_id, ticker_map, company_map, sector_map, industry_map, start=None):
    """Transform weight for dashboard DB."""
    flattened_weight = weight.loc[start:].stack().reset_index()
    flattened_weight.columns = ["base_date", "gvkey_iid", "weight"]

    flattened_weight["strategy_id"] = strategy_id
    flattened_weight["ticker"] = flattened_weight["gvkey_iid"].map(lambda e: ticker_map[e])
    flattened_weight["company"] = flattened_weight["gvkey_iid"].map(lambda e: company_map[e])
    flattened_weight["sector"] = flattened_weight["gvkey_iid"].map(lambda e: sector_map[e])
    flattened_weight["industry"] = flattened_weight["gvkey_iid"].map(lambda e: industry_map[e])

    columns = ["strategy_id", "base_date", "gvkey_iid", "ticker", "company", "sector", "industry", "weight"]
    flattened_weight = flattened_weight.loc[:, columns]

    return flattened_weight


def transform_return_for_dashboard_db(portfolio_return, strategy_id):
    """Transform return for dashboard DB."""
    if isinstance(portfolio_return, pd.Series):
        portfolio_return = portfolio_return.to_frame(name="returns").astype(np.float64)

    portfolio_return = portfolio_return.reset_index()
    portfolio_return.columns = ["base_date", "returns"]
    portfolio_return["strategy_id"] = strategy_id

    portfolio_return["cumulative_returns"] = (1 + portfolio_return["returns"]).cumprod()
    portfolio_return["log_returns"] = np.log(1 + portfolio_return["returns"])
    portfolio_return["cumulative_log_returns"] = np.log(1 + portfolio_return["cumulative_returns"])

    columns = ["strategy_id", "base_date", "returns", "cumulative_returns", "log_returns", "cumulative_log_returns"]
    portfolio_return = portfolio_return.loc[:, columns]

    return portfolio_return


if __name__ == "__main__":
    api = Kirin()
    base_path = Path(DATA_DIR) / IDENTIFIER

    # Basket1
    basket1_gvkey_iid = [e.split("_") for e in BASKET1]
    basket1_ticker_map = {"_".join(e): master_utils.retrieve_current_ticker(api, e[0], e[1]) for e in basket1_gvkey_iid}
    basket1_company_map = {"_".join(e): master_utils.retrieve_current_company(api, e[0]) for e in basket1_gvkey_iid}
    basket1_sector_map = {"_".join(e): master_utils.retrieve_current_sector(api, e[0]) for e in basket1_gvkey_iid}
    basket1_industry_map = {"_".join(e): master_utils.retrieve_current_industry(api, e[0]) for e in basket1_gvkey_iid}

    basket1_weight = pd.read_csv(base_path / BASKET1_WEIGHT_FILENAME, index_col=0, parse_dates=True)
    basket1_return = pd.read_csv(base_path / BASKET1_RETURN_FILENAME, index_col=0, parse_dates=True)

    dashboard_format_basket1_weight = transform_weight_for_dashboard_db(
        basket1_weight, "basket1", basket1_ticker_map, basket1_company_map, basket1_sector_map, basket1_industry_map, start=START_DATE
    )
    dashboard_format_basket1_return = transform_return_for_dashboard_db(
        basket1_return, "basket1"
    )
    dashboard_format_basket1_weight.to_csv(base_path / ("dashboard_format_" + BASKET1_WEIGHT_FILENAME), index=False)
    dashboard_format_basket1_return.to_csv(base_path / ("dashboard_format_" + BASKET1_RETURN_FILENAME), index=False)

    # Basket2
    basket2_gvkey_iid = [e.split("_") for e in BASKET2]
    basket2_ticker_map = {"_".join(e): master_utils.retrieve_current_ticker(api, e[0], e[1]) for e in basket2_gvkey_iid}
    basket2_company_map = {"_".join(e): master_utils.retrieve_current_company(api, e[0]) for e in basket2_gvkey_iid}
    basket2_sector_map = {"_".join(e): master_utils.retrieve_current_sector(api, e[0]) for e in basket2_gvkey_iid}
    basket2_industry_map = {"_".join(e): master_utils.retrieve_current_industry(api, e[0]) for e in basket2_gvkey_iid}

    basket2_weight = pd.read_csv(base_path / BASKET2_WEIGHT_FILENAME, index_col=0, parse_dates=True)
    basket2_return = pd.read_csv(base_path / BASKET2_RETURN_FILENAME, index_col=0, parse_dates=True)

    dashboard_format_basket2_weight = transform_weight_for_dashboard_db(
        basket2_weight, "basket2", basket2_ticker_map, basket2_company_map, basket2_sector_map, basket2_industry_map, start=START_DATE
    )
    dashboard_format_basket2_return = transform_return_for_dashboard_db(
        basket2_return, "basket2"
    )
    dashboard_format_basket2_weight.to_csv(base_path / ("dashboard_format_" + BASKET2_WEIGHT_FILENAME), index=False)
    dashboard_format_basket2_return.to_csv(base_path / ("dashboard_format_" + BASKET2_RETURN_FILENAME), index=False)
