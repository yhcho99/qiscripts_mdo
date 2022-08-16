import sys
sys.path.append("/home/hyungyunjeon/QRAFT/git_clones/qiscripts")
sys.path.append("/home/sronly/Projects/qiscripts")
from paths import DATA_DIR
from pathlib import Path
from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import final_scores
from strategy_simulation.helper import picks, weights, final_scores
NEPTUNE_TOKEN = None

IDENTIFIER = 'Plain_Vanilla_mixedHDIV_test02'
BENCHMARK = ["S&PCOMP"]

# IDENTIFIER = 'SIMULATION_QRFT'
# BENCHMARK = ["S&PCOMP", "NASA100"]
#"S&PCOMP"
#"U:MTUM"
#"NASA100"

date_dict = {
    "date_from": "2009-12-31",
    "date_to": "2021-10-31",
    "rebalancing_terms": "M",
}

setting_params = {
    "identifier": "Plain_Vanilla_mixedHDIV_test02",  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON"],
        "backtest_mode": True,
        "except_no_isin_code": False,
        "class_a_only": True,
        "primary_issue": True,
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

neptune_dict = {
    "use_neptune": False,
    "user_id": "qrft",    "project_name": "qrft",
    "exp_id": "qrft-0228",
    "exp_name": "qrft",
    "description": "qrft",
    "tags": ["qrft"],
    "hparams": {},
    "token": NEPTUNE_TOKEN,
}

st = Strategy(
        kirin_config=setting_params['kirin_config'],
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        date_from=date_dict['date_from'],
        date_to=date_dict['date_to'],
        rebalancing_terms=date_dict['rebalancing_terms'],
        long_picking_config=picks.picking_by_signal("mu/variance**0.5", False, 1, 50, ascending=False),
        long_weighting_config=(
            weights.market_weight(),
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
        factor=True,
        gics=True,
    )
st.backtest()

cp = Comparison(
    data_dir=DATA_DIR,
    identifier=setting_params['identifier'],
    kirin_config=setting_params['kirin_config'],
    performance_fps=list(st.get_performance_fps()),
    performance_names=list(st.get_performance_names()),
    standard_benchmarks=["S&PCOMP"],
    comparison_periods=[],
    final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
    neptune_params=neptune_dict
)

cp.compare()
