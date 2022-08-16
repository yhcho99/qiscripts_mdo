import copy
from qraft_data.universe import Universe
from utils.load_data import LoadData
from product.master.baskets import *


IT_CODE = 45



def get_base_filter(loader: LoadData):
    base_filter = loader.call_if_not_loaded("price", ["high_level", "equity", "get_monthly_price_data"])
    base_filter._data = base_filter._data.notna()

    return base_filter


def get_basket2_filter(base_filter):
    infer_columns = base_filter.columns
    unselected_keys = infer_columns[~infer_columns.isin(BASKET2)]

    basket2_filter = base_filter.copy()
    basket2_filter.loc[:, unselected_keys] = False

    return basket2_filter


def get_nasdaq_filter(base_filter, default_config, cache_path, dataset_path):
    nasdaq_config = copy.deepcopy(default_config)
    nasdaq_config["cache_dir"] = cache_path
    nasdaq_config["exchange"] = ["NASDAQ"]

    nasdaq_universe = Universe(**nasdaq_config)
    nasdaq_loader = LoadData(path=dataset_path, universe=nasdaq_universe)

    nasdaq_filter = nasdaq_loader.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    nasdaq_filter._data = nasdaq_filter._data.reindex(base_filter.columns, axis=1)
    nasdaq_filter._data = nasdaq_filter._data.notna()

    return nasdaq_filter


def get_nasdaq_top100_filter(base_filter, default_config, cache_path, dataset_path):
    nasdaq_config = copy.deepcopy(default_config)
    nasdaq_config["cache_dir"] = cache_path
    nasdaq_config["exchange"] = ["NASDAQ"]

    nasdaq_universe = Universe(**nasdaq_config)
    nasdaq_loader = LoadData(path=dataset_path, universe=nasdaq_universe)

    top100_filter = nasdaq_loader.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    top100_filter._data = top100_filter._data.reindex(base_filter.columns, axis=1)
    top100_filter._data = top100_filter._data[base_filter._data]
    top100_filter = top100_filter.rank(ascending=False, pct=False) <= 100

    return top100_filter


def get_nyse_nasdaq_top500_filter(base_filter, loader: LoadData):
    top500_filter = loader.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    top500_filter._data = top500_filter._data[base_filter._data]
    top500_filter = top500_filter.rank(ascending=False, pct=False) <= 500
    
    return top500_filter


def get_it_filter(base_filter, loader: LoadData):
    it_filter = loader.call_if_not_loaded("sector", ["compustat", "get_historical_gics"], kwargs={"gics_level": "sector"})
    it_filter._data = it_filter._data.reindex(base_filter.columns, axis=1)
    it_filter._data = it_filter._data == IT_CODE

    return it_filter


def get_it_top100_filter(baes_filter, loader: LoadData):
    sector = loader.call_if_not_loaded("sector", ["compustat", "get_historical_gics"], kwargs={"gics_level": "sector"})
    market_value = loader.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])

    sector._data = sector._data.reindex(baes_filter.columns, axis=1)
    market_value._data = market_value._data.reindex(baes_filter.columns, axis=1)
    market_value._data = market_value._data[sector._data == IT_CODE]
    it_top100_filter = market_value.rank(ascending=False, pct=False) <= 100

    return it_top100_filter


def get_it_top60_filter(baes_filter, loader: LoadData):
    sector = loader.call_if_not_loaded("sector", ["compustat", "get_historical_gics"], kwargs={"gics_level": "sector"})
    market_value = loader.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])

    sector._data = sector._data.reindex(baes_filter.columns, axis=1)
    market_value._data = market_value._data.reindex(baes_filter.columns, axis=1)
    market_value._data = market_value._data[sector._data == IT_CODE]
    it_top100_filter = market_value.rank(ascending=False, pct=False) <= 60

    return it_top100_filter
