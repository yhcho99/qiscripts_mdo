import sys
sys.path.append("")
import numpy as np
from pathlib import Path
from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores
from paths import DATA_DIR, NEPTUNE_TOKEN

from strategy_simulation.strategy.weighting import _add_mw_tw_ew



#################################### 공통설정부분 ##########################################

IDENTIFIER = "master_jh_qrft_weekly_test"
VER = 1
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}_{VER}"

DATE_FROM = "2021-02-01"
DATE_TO = "2021-04-30"
REBAL_TERM = 'W-FRI'

EPHOCS = 10
TRAINING_LENGTH = 36
SAMPLE_LENGTH = 12


#################################### simulation ##########################################

def custom_weight(dt, signal, kirin_config, mv, tv):
    if len(signal) == 1:
        z = signal
    else:
        z = (signal - signal.mean()) / (signal.std() + 1e-6)
    adj =  0 + 1 / (1 + np.exp(-z))
    # sig = _add_mw_tw_ew(dt, adj, kirin_config, mv, tv)

    w = adj
    w = w / w.sum()
    return w.sort_values(ascending=False)


strategy_dict = {
    "from_which": "infer",  # infer: infer 데이터로부터 strategy가 진행됩니다. portfolio : universe.csv와 weight.csv가 존재할 경우입니다
    
    "short": False,  # short을 할지의 여부입니다
    "short_amount": 0.0,  # short포트폴리오의 총 비중입니다. ex) longonly의 경우 0. , 130/30의 경우 -0.3
    "short_picking_config": picks.picking_by_signal("out", False, 1, None, ascending=True),  # 숏포트폴리오 뽑는방법
    "short_weighting_config": weights.market_weight(),  # 숏 종목 비중 주는 방법

    "long_amount": 1.0,  # long포트폴리오의 총 비중입니다. ex) longonly의 경우 1.0  , 130/30의 경우 1.3,
    "long_picking_config": picks.picking_by_signal("out", False, 1, 50, ascending=False),  # 롱포트폴리오 뽑는방법 제한조건이 없을경우 None
    "long_weighting_config": (
        {'name': 'custom_w', 'custom_weighting_func': custom_weight},
    ),

    ########### 이번 실험에서는 구할때는 아래를 따로 설정해줄 필요가 없습니다 ############
    "weight_adjusting_unitdate": False,
    "backtest_daily_out": False,
    "backtest_daily_out_lag": [0, 1],
    "factor": False,
    "save_factor": False,
    "market_percentile": 0.2,
    "gics": False,
    "save_gics": False,
    "gics_level": ["sector"],
}


comparison_dict = {
    "performance_fps": [],
    "performance_names": [],
    "standard_benchmarks": [],
    "comparison_periods": [],
    "final_score": final_scores.annualized_return(
        exponential_decay_rate=None, total=False
    ),
}

neptune_dict = {
    "use_neptune": True,  # 테스트기간동안에는 잠시 False로 해두겠습니다.
    "user_id": "jayden",
    "project_name": "jayden",

    "exp_name": "실험의 이름을 설정합니다",  # 필수는 아님
    "description": "실험의 설명을 작성합니다",  # 필수는 아님

    "tags": [],  # 마스터 프로젝트에서는 변경하지 않습니다
    "exp_id": ['NEW'],  # 마스터 프로젝트에서는 변경하지 않습니다
    "hparams": {**strategy_dict,},
    "token": NEPTUNE_TOKEN,
}

setting_params = {
    "identifier": IDENTIFIER,  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "jh_master_cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ", "OTC"],
        "security_type": ["COMMON", "ADR"],
        "backtest_mode": True,
        "except_no_isin_code": False,
        "class_a_only": False,
        "pretend_monthend": False,
        "frequency" :'Weekly'
    },
    "seed_config": {
        "training_model": True,
        "training_data_loader": None,
        "train_valid_data_split": None,
    },
    "csv_summary_save": True,
    "omniboard_summary_save": False,
    "tensorboard_summary_save": True,

    "cpu_count": 8
}

date_dict = {
    "date_from": DATE_FROM,
    "date_to": DATE_TO,
    "rebalancing_terms": REBAL_TERM,
}

if __name__ == "__main__":

    st = Strategy(
        data_dir=DATA_DIR,
        setting_params=setting_params,
        date_params=date_dict,
        **strategy_dict,
    )

    st.backtest()

    comparison_dict["performance_fps"] += st.get_performance_fps()
    comparison_dict["performance_names"] += st.get_performance_names()

    p_pfs_list = comparison_dict["performance_fps"].copy()
    p_names_list = comparison_dict["performance_names"].copy()

    UNIVERSE_LIST = [
        "UNIVERSE1_MCAP_WEEKLY",
        "UNIVERSE2_MCAP_WEEKLY",
        "UNIVERSE3_MCAP_WEEKLY",
    ]

    INFER_LIST = [ 'U1', 'U2', 'U3']

    for i in range(len(INFER_LIST)):
        comparison_dict["performance_fps"] = [p_pfs_list[i]]
        comparison_dict["performance_names"] = [p_names_list[i]]
        comparison_dict["standard_benchmarks"] = [UNIVERSE_LIST[i]]

        neptune_dict["tags"] = [NEPTUNE_IDENTIFIER] + [INFER_LIST[i]]

        cp = Comparison(
            data_dir=DATA_DIR,
            setting_params=setting_params,
            neptune_params=neptune_dict,
            **comparison_dict,
        )
        cp.compare()












