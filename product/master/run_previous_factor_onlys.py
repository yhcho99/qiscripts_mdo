import sys
import time
from pathlib import Path
import copy

sys.path.append(".")

import pandas as pd
import torch.multiprocessing as mp

from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from qraft_data.data import Tag as QraftDataTag

import batch_maker as bm

from strategy_integration.components.integration.deep.deep_integration import DeepIntegration
from strategy_integration.components.models.deep_models.multi_source_attention_model import MultiSourceAttentionModel

from utils.load_data import LoadData
from utils.multiprocess import MultiProcess

from paths import DATA_DIR, NEPTUNE_TOKEN
from product.master.baskets import *
from product.master.five_factors import LoadFiveFactor
import product.master.infer_filters as infer_filters


EPOCHS = 23
BATCH_SIZE = 32
CROSS_NUM_ITER = 100
CROSS_NUM_UNIVERSE = 256
SAMPLE_LENGTH = 36

# ROOT = "master_test_traditional_factors_only_5years"
# IDENTIFIER = ROOT
# CACHE_PATH = f"cache_master_test_traditional_factors_only_5years"
# DATASET_PATH = f"dataset_master_test_traditional_factors_only_5years"
# NASDAQ_CACHE_PATH = f"cache_master_test_traditional_factors_only_5years_nasdaq_only"
# NASDAQ_DATASET_PATH = f"dataset_master_test_traditional_factors_only_5years_nasdaq_only"
# NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

# ROOT = "master_test_traditional_factors_only_3years"
# IDENTIFIER = ROOT
# CACHE_PATH = f"cache_master_test_traditional_factors"
# DATASET_PATH = f"dataset_master_test_traditional_factors"
# NASDAQ_CACHE_PATH = f"cache_master_test_traditional_factors_nasdaq_only"
# NASDAQ_DATASET_PATH = f"dataset_master_test_traditional_factors_nasdaq_only"
# NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

ROOT = "master_test_traditional_factors_only_1years"
IDENTIFIER = ROOT
CACHE_PATH = f"cache_master_test_traditional_factors"
DATASET_PATH = f"dataset_master_test_traditional_factors"
NASDAQ_CACHE_PATH = f"cache_master_test_traditional_factors_nasdaq_only"
NASDAQ_DATASET_PATH = f"dataset_master_test_traditional_factors_nasdaq_only"
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"


DATE_FROM = "2009-12-31"
DATE_TO = "2022-01-31"

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
]

neptune_dict = {
    "use_neptune": False,  # 넵튠을 사용하려면 True 로 표시합니다.
    "user_id": "jayden",  # USER_ID는 당분간 저로 고정합니다
    "project_name": "jayden",  # 프로젝트는 jayden, tei, aiden 으로 만들어 뒀습니다
    "exp_name": "",  # 실험의 이름 필수는 아닙니다
    "description": "",  # 실험의 설명 필수는 아닙니다
    "hparams": {},  # 저장하고 싶은 하이퍼 파라미터를 딕셔너리로, 굳이 안넣어줘도 소스코드가 저장되어 당시의 셋팅을 확인가능합니다.
    "tags": [],  # 마스터 프로젝트에서는 태그를 변경하지 않습니다
    "exp_id": ["NEW"],  # 마스터 프로젝트에서는 EXP_ID를 변경하지 않습니다
    "token": NEPTUNE_TOKEN,  # 키는 고정입니다
}

setting_params = {
    "identifier": IDENTIFIER,  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": str(Path(DATA_DIR) / CACHE_PATH),
        "use_sub_server": True,
        "exchange": ["NYSE", "NASDAQ"],
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

integrated_model_params = {
    "training_all_at_once": False,
    "reinitialization": 1,
    "models": [
        {
            "name": MultiSourceAttentionModel.__name__,
            "inputs": ["x"],
            "targets": ["y"],
            "forwards": [],
            "cross_units": {
                "x": [64, 32],
            },
            "batch_units": [32, 16],
            "dropout": 0.0,
            "layer_norm": True,
            "noise": 0.01,
            "learning_rate": 3e-4,
            "weight_decay": 3e-5,
            "huber_loss_positive_beta": 1.0,
            "huber_loss_negative_beta": 1.0,
            "stage": (10, 20),
        },
    ],
    "forwards_at_epoch": [],
    "forward_names": [],
    "early_stopping_start_after": 100,
    "early_stopping_interval": 10,
    "epochs": EPOCHS,
    "path": DATA_DIR,
    "save_all_epochs": False,
    "weight_share": False,
    "batch_update": ["optimizer"],
    "epoch_update": [],
    "batch_size": BATCH_SIZE,
    "validation_ratio": 0.4,
    "shuffle": True,
}

def run_model(_, context_name_list, context_list, dict_of_shapes, inference_date):
    # DeepIntegration instance 생성
    di = DeepIntegration(
        identifier=setting_params["identifier"],
        sub_identifier=inference_date.strftime("%Y-%m-%d"),
        dataset=None,
        data_shape=dict_of_shapes,
        integrated_model_params=integrated_model_params,
        setting_params=setting_params,
    )
    di.initialize_train()
    di.saver.restore(step=23)
    # context list를 만듬
    inference_lc_list = context_list
    
    for i, model in enumerate(di.models):
        model.net = model.net.cuda(model.device)

    # early stopping 객체 초기화
    for context_name, lc in zip(context_name_list, inference_lc_list):
        for meta, data in lc.inference_iterate(cross_sampling=True):
            sec_keys = meta["SECURITY"][0]

            di.infer_with_all_epochs(context_name, data, sec_keys)


if __name__ == "__main__":
    universe = Universe(**setting_params["kirin_config"])
    api = get_kirin_api(universe)

    master_dataset_fp = Path(DATA_DIR) / DATASET_PATH
    loader = LoadData(path=master_dataset_fp, universe=universe)
    with mp.Pool(4) as pool:
        raw_inputs = pool.starmap(loader.call_if_not_loaded, input_data)

    inputs = []
    for raw_data in raw_inputs:
        if raw_data.get_tag() == QraftDataTag.INDEX.value:
            raw_data = raw_data.rolling(SAMPLE_LENGTH).zscore()
        if raw_data.get_tag() == QraftDataTag.EQUITY.value:
            raw_data = 2 * raw_data.rank(pct=True) - 1
        inputs.append(raw_data)

    # preprocess mv & output
    y = loader.call_if_not_loaded("tr_1m_0m", ["high_level", "equity", "get_monthly_price_return"], [1, 0])
    y = 2 * y.rank(pct=True) - 1

    # training filter
    training_filter = infer_filters.get_base_filter(loader)
    basket2_filter = infer_filters.get_basket2_filter(training_filter)
    nasdaq_top100_filter = infer_filters.get_nasdaq_top100_filter(
        training_filter, 
        setting_params["kirin_config"], 
        str(Path(DATA_DIR) / NASDAQ_CACHE_PATH), 
        str(Path(DATA_DIR) / NASDAQ_DATASET_PATH)
    )
    nyse_nasdaq_top500_filter = infer_filters.get_nyse_nasdaq_top500_filter(training_filter, loader)
    it_all_filter = infer_filters.get_it_filter(training_filter, loader)
    it_top100_filter = infer_filters.get_it_top100_filter(training_filter, loader)
    it_top60_filter = infer_filters.get_it_top60_filter(training_filter, loader)

    probability = training_filter.copy()
    probability._data = probability._data.astype(float)

    # inference filter
    infer_filter_list = [basket2_filter, nasdaq_top100_filter, nyse_nasdaq_top500_filter, it_all_filter, it_top100_filter, it_top60_filter]
    name_list = ["BASKET2", "NASDAQ_TOP100" , "NYSE_NASDAQ_TOP500", "IT_ALL", "IT_TOP100", "IT_TOP60"]
    
    # Check input, output and filter dataset has same index and columns
    index, columns = bm.checking.check_equal_index_and_columns(inputs + infer_filter_list + [training_filter])

    sbm_list = []
    for name, infer_filter in zip(name_list, infer_filter_list):
        binder_x = bm.DataBinderV3(
            data_list=inputs,
            training_filter=training_filter,
            inference_filter=infer_filter,
            length=SAMPLE_LENGTH,
            is_input=True,
            max_nan_ratio=1.0,
            aug=None,
        )
        binder_y = bm.DataBinderV3(
            data_list=[y],
            training_filter=training_filter,
            inference_filter=infer_filter,
            length=1,
            is_input=False,
            max_nan_ratio=0.0,
            aug=None,
        )
        sbm = bm.SecurityBatchMaker(
            save_path=str(Path(DATA_DIR) / setting_params["identifier"] / "batch_dataset" / name),
            index=index,
            columns=columns,
            data_map={"x": binder_x, "y": binder_y},
            max_cache="30GB",
            probability=probability,
        )
        sbm_list.append(sbm)
    shapes_dict = sbm_list[0].get_sample_shapes()

    date_list = pd.date_range(
        start=date_dict["date_from"],
        end=date_dict["date_to"],
        freq=date_dict["rebalancing_terms"],
    ).to_pydatetime()
    
    for d_inference in date_list:
        lc_list = []
        length_list = [0] + [0] * (len(sbm_list) - 1)
        for sbm, length in zip(sbm_list, length_list):
            lc = sbm.local_context(inference_dates=d_inference, length=length)
            lc_list.append(lc)
        
        run_model("_", name_list, lc_list, shapes_dict, d_inference)
