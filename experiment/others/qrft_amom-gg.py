import sys
import time
sys.path.append("")
import pandas as pd
import batch_maker as bm
import multiprocessing as mp

from pathlib import Path
from paths import DATA_DIR, NEPTUNE_TOKEN
from utils.multiprocess import MultiProcess
from utils.load_data import LoadData

from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from qraft_data.data import Tag as QraftDataTag

from strategy_integration.components.integration.deep.deep_integration import (
    DeepIntegration,
)
from strategy_integration.components.models.deep_models import attention_model

from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores


#################################### 공통설정부분 ##########################################
setting_params = {
    "identifier": "aaaaabbbbb",  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON"],
        "backtest_mode": True,
        "except_no_isin_code": False,
        "class_a_only": True,
        "primary_issue":True,
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

    "cpu_count": 4
}

date_dict = {
    "date_from": "2014-12-31",
    "date_to": "2020-08-31",
    "rebalancing_terms": "M",
}


neptune_dict = {
    "use_neptune": False,  # 테스트기간동안에는 잠시 False로 해두겠습니다.
    "user_id": "qrft",
    "project_name": "qrft",
    "exp_id": "qrft-0228",
    "exp_name": "qrft",
    "description": "qrft",
    "tags": ["qrft"],
    "hparams": {},
    "token": NEPTUNE_TOKEN,
}



if __name__ == "__main__":
    st = Strategy(
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        kirin_config=setting_params['kirin_config'],
        date_from=date_dict['date_from'],
        date_to=date_dict['date_to'],
        rebalancing_terms=date_dict['rebalancing_terms'],

        long_picking_config=picks.picking_by_signal("out", False, 1, 50, ascending=False),
        long_weighting_config=(
            weights.dynamic_weight("mw"),
            # weights.rank_sum_discounted_weight(0.995, 50, 0.5),
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
        backtest_daily_out=False,
        backtest_daily_out_lag=[1],

        # factor=True,
        # market_percentile=0.2,
        #
        # gics=True,
        # gics_level=["sector"],
    )

    st.backtest()

    cp = Comparison(
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        kirin_config=setting_params['kirin_config'],
        performance_fps=list(st.get_performance_fps()),
        performance_names=list(st.get_performance_names()),
        standard_benchmarks=["U:MTUM","U:SPMO"],
        comparison_periods=[],
        final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
        neptune_params=neptune_dict
    )

    cp.compare()
