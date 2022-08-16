import sys
import time
import warnings

sys.path.append("")
import pandas as pd
import numpy as np
import scipy.optimize
import batch_maker as bm
import torch.multiprocessing as mp

from pathlib import Path
from paths import DATA_DIR, NEPTUNE_TOKEN
from utils.multiprocess import MultiProcess
from utils.load_data import LoadData
from qraft_data.util import get_kirin_api
from utils.ff_manager import FFManager
from qraft_data.universe import Universe
from qraft_data.data import Tag as QraftDataTag
from strategy_integration.components.integration.deep.deep_integration import DeepIntegration
from qraft_data.data import QraftData

from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores
from strategy_simulation.strategy.weighting import _add_mw_tw_ew
from experiment.master_phase1.masters.multisource_attention_model import TestModel
from utils.master_phase2_basket_gvkey import *


EPOCHS = 23
BATCH_SIZE = 32
CROSS_NUM_ITER = 100
CROSS_NUM_UNIVERSE = 256
TRAINING_LENGTH = 48
SAMPLE_LENGTH = 36
ACT_CF = 0.99
VAR_CF = 0.9999

IDENTIFIER = "parallel_download_test"# f"master_phase2-ff+wo_alt-try2"
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

DATE_FROM = "2000-12-31"
DATE_TO = "2021-09-30"


setting_params = {
    "identifier": IDENTIFIER,  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "parallel_download_testcahce").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ", "OTC"],
        "security_type": ["COMMON", "ADR"],
        "backtest_mode": True,
        "except_no_isin_code": False,
        "class_a_only": False,
        "primary_issue": False,
        "pretend_monthend": False,
    },
    "seed_config": {
        "training_model": True,
        "training_data_loader": None,
        "train_valid_data_split": None,
    },
    "csv_summary_save": True,
    "omniboard_summary_save": False,
    "tensorboard_summary_save": True,
    "cpu_count": 12,
}

date_dict = {
    "date_from": DATE_FROM,
    "date_to": DATE_TO,
    "rebalancing_terms": "M",
}

# Preprocess input dataset
input_data = [
    ("pr_1m_0m", ["high_level", "equity", "get_monthly_price_return"], [1, 0]),
    ("mv", ["high_level", "equity", "get_monthly_market_value"]),
    ("btm", ["high_level", "equity", "get_book_to_market"]),
    ("mom_12m_1m", ["high_level", "equity", "get_monthly_price_return"], [12, 1]),
    ("ram_12m_0m", ["high_level", "equity", "get_ram"], ["pi", 12]),
    ("vol_3m", ["high_level", "equity", "get_monthly_volatility"], [3]),
    ("res_mom_12m_1m_0m", ["high_level", "equity", "get_res_mom"], [12, 1, 0]),
    ("res_vol_6m_3m_0m", ["high_level", "equity", "get_res_vol"], [6, 3, 0]),
    ("at", ["high_level", "equity", "get_asset_turnover"]),
    ("gpa", ["compustat", "custom_api", "get_gpa"]),
    ("rev_surp", ["high_level", "equity", "get_revenue_surprise"]),
    ("cash_at", ["high_level", "equity", "get_cash_to_asset"]),
    ("op_lev", ["high_level", "equity", "get_operating_leverage"]),
    ("roe", ["high_level", "equity", "get_roe"]),
    ("std_u_e", ["high_level", "equity", "get_standardized_unexpected_earnings"]),
    ("ret_noa", ["high_level", "equity", "get_return_on_net_operating_asset"]),
    ("etm", ["high_level", "equity", "get_earnings_to_market"]),
    ("ia_mv", ["high_level", "equity", "get_linear_assumed_intangible_asset_to_market_value"],),
    ("ae_m", ["high_level", "equity", "get_advertising_expense_to_market"]),
    ("ia_ta", ["high_level", "equity", "get_linear_assumed_intangible_asset_to_total_asset"],),
    ("rc_a", ["high_level", "equity", "get_rnd_capital_to_asset"]),
    ("r_s", ["high_level", "equity", "get_rnd_to_sale"]),
    ("r_a", ["high_level", "equity", "get_rnd_to_asset"]),
    ("fred_ff", ["fred", "ff"]),
    ("t3m", ["high_level", "macro", "get_us_treasury_3m"]),
    ("t6m", ["fred", "treasury_6m"]),
    ("t2y", ["high_level", "macro", "get_us_treasury_2y"]),
    ("t3y", ["fred", "treasury_3y"]),
    ("t5y", ["high_level", "macro", "get_us_treasury_5y"]),
    ("t7y", ["fred", "treasury_7y"]),
    ("t10y", ["high_level", "macro", "get_us_treasury_10y"]),
    ("aaa", ["fred", "aaa_corporate"]),
    ("baa", ["fred", "baa_corporate"]),
    ("snp500_pr", ["high_level", "index", "get_snp500_momentum"], [], {"periods": 1}),
    ("wilshire500_pr", ["high_level", "index", "get_wilshire5000_price_index_momentum"], [], {"periods": 1},),
    ("ted", ["fred", "ted_spread"]),
    ("t1y_ff", ["fred", "t1y_minus_ff"]),
    ("t5y_ff", ["fred", "t5y_minus_ff"]),
    ("t10y_t2y", ["fred", "t10y_minus_t2y"]),
    ("aaa_t10y", ["fred", "aaa_minus_t10y"]),
    ("baa_t10y", ["fred", "baa_minus_t10y"]),
    ("aaa_ff", ["fred", "aaa_minus_ff"]),
    ("baa_ff", ["fred", "baa_minus_ff"]),
    ("core_cpi", ["high_level", "macro", "get_core_cpi_rate"]),
    ("core_pce", ["high_level", "macro", "get_core_pce_rate"]),
    ("core_ppi", ["high_level", "macro", "get_core_ppi_rate"]),
    ("cpi", ["high_level", "macro", "get_cpi_rate"]),
    ("pce", ["high_level", "macro", "get_pce_rate"]),
    ("ppi", ["high_level", "macro", "get_ppi_rate"]),
    ("trimmed_pce", ["high_level", "macro", "get_trimmed_mean_pce_rate"], [], {"change_to": "none"},),
    ("unemploy", ["high_level", "macro", "get_unemployment_rate"]),
    #  ("retail_mfr", ["high_level", "macro", "get_retail_money_funds_rate"]),
    ("m1", ["high_level", "macro", "get_us_m1_rate"]),
    ("m2", ["high_level", "macro", "get_us_m2_rate"]),
    ("export_growth", ["fred", "exports_growth"]),
    ("import_growth", ["fred", "imports_growth"]),
    ("real_gig", ["fred", "real_government_investment_growth"]),
    ("real_pig", ["fred", "real_private_investment_growth"]),
    ("federal_tg", ["high_level", "macro", "get_federal_government_current_tax_receipts_growth"],),
    ("real_gdp", ["fred", "real_gdp_growth"]),
    ("corporate_tg", ["high_level", "macro", "get_corporate_profits_after_tax_growth"]),
    ("industrial_prod", ["high_level", "macro", "get_industrial_production_index_rate"],),
    ("home_pr", ["high_level", "macro", "get_home_price_index_rate"]),
    ("wti", ["high_level", "macro", "get_wti_price_rate"]),
    ("capa_util", ["fred", "capacity_utilization"]),
    ("snp500_pe", ["high_level", "macro", "get_snp_pe"]),
    ("snp500_vol", ["high_level", "macro", "get_snp500_vol"]),
]


if __name__ == "__main__":
    universe = Universe(**setting_params["kirin_config"])
    import pdb; pdb.set_trace();
    api = get_kirin_api(universe)

#     from kirin import Kirin
#     config = setting_params["kirin_config"].copy()
#     cache_dir = config.pop("cache_dir")
#     use_sub_server = config.pop("use_sub_server")
#     pretend_monthend = config.pop("pretend_monthend")
#     config["class_A_only"] = config.pop("class_a_only")

#     _ = config.pop("backtest_mode"), config.pop("primary_issue")
#     _ = Kirin(
#         cache_dir=cache_dir, 
#         use_sub_server_for_compustat=use_sub_server, 
#         pretend_monthend=pretend_monthend
#     ).compustat.set_investment_universe(**config)

    master_dataset_fp = Path(DATA_DIR) / "parallel_download_testset"
    load_dataset = LoadData(path=master_dataset_fp, universe=universe)
    with mp.Pool(8) as pool:
        raw_inputs = pool.starmap(load_dataset.call_if_not_loaded, input_data)
