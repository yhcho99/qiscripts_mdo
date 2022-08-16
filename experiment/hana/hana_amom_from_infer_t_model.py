import sys
import time
import warnings
import argparse
sys.path.append("")
import pandas as pd
import numpy as np
import scipy.optimize

from pathlib import Path
from paths import DATA_DIR, NEPTUNE_TOKEN


from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores
from strategy_simulation.strategy.weighting import _add_mw_tw_ew
from experiment.masters.jh_model import TestModel

###################################################################################################
############################################# 설정부분 ##############################################
###################################################################################################


# parser = argparse.ArgumentParser()
# parser.add_argument('--cf', type=float)
# parser.add_argument('--var', type=float)
# parser.add_argument('--months', type=str)


# args = parser.parse_args()
# ARGS_CF = args.cf
# ARGS_VAR = args.var
# ARGS_MONTH = args.months

MODEL=  'AMOM'
NVQ_CF=  0.1
AMOM_CF= 1 - NVQ_CF
MV_MASK= 100
MASK=    50

#################################### 공
IDENTIFIER =  f"BTS_model_{MODEL}_NVQ{int(NVQ_CF*100):2d}_AMOM{int(AMOM_CF*100):2d}_MV{MV_MASK}_FILTER{MASK}_Short"
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

DATE_FROM = "2016-07-31"
DATE_TO = "2021-08-31"

comparison_dict = {
    "performance_fps": ['BASE_FILTER'],  # identifier와 동일한 값 혹은, 전체 performance file paths
    "performance_names": ['BASE_FILTER'],  # 각 퍼포먼스 별 별칭
    "standard_benchmarks": ["@QQQ"],  # 벤치마크로 삼을 U:SPY
    "comparison_periods": [],  # 비교하고 싶은 기간
    "final_score": final_scores.annualized_return(exponential_decay_rate=None, total=False),
}

neptune_dict = {
    "use_neptune": False,  # 넵튠을 사용하려면 True 로 표시합니다.
    "user_id": "jayden",  # USER_ID는 당분간 저로 고정합니다
    "project_name": "jayden",  # 프로젝트는 jayden, tei, aiden 으로 만들어 뒀습니다
    "exp_name": "",  # 실험의 이름 필수는 아닙니다
    "description": "",  # 실험의 설명 필수는 아닙니다
    "hparams": {},  # 저장하고 싶은 하이퍼 파라미터를 딕셔너리로,
    "tags": [],  # 마스터 프로젝트에서는 태그를 변경하지 않습니다
    "exp_id": ["NEW"],  # 마스터 프로젝트에서는 EXP_ID를 변경하지 않습니다
    "token": NEPTUNE_TOKEN,  # 키는 고정입니다
}

###################################################################################################
###################################################################################################
setting_params = {
    "identifier": IDENTIFIER,  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
        "use_sub_server": True,
        "exchange": ["NYSE"],
        "security_type": ["COMMON"],
        "backtest_mode": True,
        "except_no_isin_code": False,
        "class_a_only": False,
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

if __name__ == "__main__":
    st = Strategy(
        kirin_config=setting_params['kirin_config'],
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        date_from=date_dict['date_from'],
        date_to=date_dict['date_to'],
        rebalancing_terms=date_dict['rebalancing_terms'],
        long_picking_config=picks.picking_by_signal("mu", False, 1, 50, ascending=False),
        long_weighting_config=(
            weights.dynamic_weight("0.5*mw + 0.5*ew"),
            weights.optimal_weight(
                kirin_config=setting_params["kirin_config"],
                loss_type="MSE",
                max_weight=0.08,
                threshold_weight=0.05,
                bound_sum_threshold_weight=0.4,
                bound_gics={"sector": 0.5, "industry": 0.24},
                bound_financials_sector={"40": 0.048},
            ),
        ),
    )
    st.backtest()

    comparison_dict["performance_fps"] += st.get_performance_fps()
    comparison_dict["performance_names"] = st.get_performance_names()

    cp = Comparison(
        data_dir=DATA_DIR,
        setting_params=setting_params,
        neptune_params=neptune_dict,
        **comparison_dict,
    )
    cp.compare()
