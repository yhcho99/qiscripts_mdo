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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str)
parser.add_argument("--act", type=float)
parser.add_argument("--tryv", type=int)
args = parser.parse_args()


EPOCHS = 23
BATCH_SIZE = 32
CROSS_NUM_ITER = 100
CROSS_NUM_UNIVERSE = 256
TRAINING_LENGTH = 48
SAMPLE_LENGTH = 36
ACT_CF = args.act
VAR_CF = 0.9999
TRY_V = args.tryv
TARGET = args.target

IDENTIFIER = f"master_phase2-{TARGET}-try{TRY_V}-ACT_CF{ACT_CF:.4f}"
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

DATE_FROM = "2000-12-31"
DATE_TO = "2021-08-31"


def test_picking(infer: pd.DataFrame):
    data = infer.values
    num = data.shape[1]

    def negative_opt(w, factors, factor_covariance, factor_stds):
        factor_returns = (factors * w).sum(axis=1)
        expected_return = factor_returns.sum()
        factor_exposures = factor_returns / factor_stds
        var = factor_exposures.dot(factor_exposures) * (17 / 16) + 1e-10
        print(expected_return, np.sqrt(var))

        return -(expected_return - VAR_CF * var)

    def sum_is_one(x):
        return np.sum(x) - 1.0

    bounds = [(0.0, 1.0) for _ in range(num)]
    cons = {"type": "eq", "fun": sum_is_one}

    with np.errstate(divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cov = np.nan_to_num(np.corrcoef(data))

    with np.errstate(divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stds = np.nan_to_num(np.std(data, axis=1))

    weight = np.full(num, 1 / num, dtype=float)

    opts = scipy.optimize.minimize(
        negative_opt,
        weight,
        args=(data, cov, stds),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )
    if not opts.success:
        warnings.warn(f"optimal weight not optimized.")

    return pd.Series(
        opts.x / np.sum(opts.x), index=infer.columns
    ).sort_values(
        ascending=False
    )


def test_weight(dt, signal, kirin_config, mv, tv):
    df = _add_mw_tw_ew(dt, signal, kirin_config, mv, tv)
    w_sig = df["signal"]
    w_mv = df["mw"]

    act_cf, cap_cf = ACT_CF, 1 - ACT_CF
    w = act_cf * w_sig + cap_cf * w_mv
    w = w / w.sum()
    return w.sort_values(ascending=False)


strategy_dict = {
    "from_which": "infer",
    "long_amount": 1.0,
    "long_picking_config": {
        "name": "custom_picking",
        "custom_picking_func": test_picking,
    },
    "long_weighting_config": ({
        "name": "custom_weighting",
        "custom_weighting_func": test_weight
    },),

    "short": False,
    "short_amount": 0.0,
    "short_picking_config": picks.picking_by_signal(
        "out", False, 1, None, ascending=True
    ),
    "short_weighting_config": weights.market_weight(),
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
    "use_neptune": False,  # 넵튠을 사용하려면 True 로 표시합니다.
    "user_id": "jayden",  # USER_ID는 당분간 저로 고정합니다
    "project_name": "jayden",  # 프로젝트는 jayden, tei, aiden 으로 만들어 뒀습니다
    "exp_name": "",  # 실험의 이름 필수는 아닙니다
    "description": "",  # 실험의 설명 필수는 아닙니다
    "hparams": {**strategy_dict},  # 저장하고 싶은 하이퍼 파라미터를 딕셔너리로, 굳이 안넣어줘도 소스코드가 저장되어 당시의 셋팅을 확인가능합니다.
    "tags": [],  # 마스터 프로젝트에서는 태그를 변경하지 않습니다
    "exp_id": ["NEW"],  # 마스터 프로젝트에서는 EXP_ID를 변경하지 않습니다
    "token": NEPTUNE_TOKEN,  # 키는 고정입니다
}

setting_params = {
    "identifier": IDENTIFIER,  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "master_phase2_cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ", "OTC"],
        "security_type": ["COMMON", "ADR"],
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
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        kirin_config=setting_params['kirin_config'],
        date_from=date_dict['date_from'],
        date_to=date_dict['date_to'],
        rebalancing_terms=date_dict['rebalancing_terms'],

        long_picking_config={
            "name": "custom_picking",
            "custom_picking_func": test_picking,
        },
        long_weighting_config={
            "name": "custom_weighting",
            "custom_weighting_func": test_weight
        },
        factor=False,
        market_percentile = 0.35,  # 시가총액 상위 몇 %의 주식을 볼지

        gics=False,
        gics_level=["sector"]
    )

    st.backtest()

    cp = Comparison(
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        kirin_config=setting_params['kirin_config'],
        performance_fps=list(st.get_performance_fps()),
        performance_names=list(st.get_performance_names()),
        standard_benchmarks=["MASTER2_BASKET1", "MASTER2_BASKET2"],
        comparison_periods=[],
        final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
        neptune_params=neptune_dict
    )

    cp.compare()
