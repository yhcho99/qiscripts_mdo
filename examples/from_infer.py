from paths import DATA_DIR
from pathlib import Path

from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import final_scores
from strategy_simulation.helper import picks, weights, final_scores
NEPTUNE_TOKEN = None

IDENTIFIER = 'paradigm_test_quarterly'
BENCHMARK = ["S&PCOMP"]

# IDENTIFIER = 'SIMULATION_QRFT'
# BENCHMARK = ["S&PCOMP", "NASA100"]
#"S&PCOMP"
#"U:MTUM"
#"NASA100"

kirin_config = {
    "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
    "use_sub_server": False,
    "exchange": ["NYSE", "NASDAQ"],
    "security_type": ["COMMON"],
    "backtest_mode": True,
    "except_no_isin_code": False,
    "class_a_only": False,
    "primary_issue": True,
    "pretend_monthend": False,
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
    data_dir=DATA_DIR,
    identifier=IDENTIFIER,
    kirin_config=kirin_config,

    date_from='2010-12-31',
    date_to='2021-09-30',
    rebalancing_terms='3M',

    long_picking_config=picks.picking_by_signal("out", False, 1, 50, ascending=False),
    long_weighting_config=(
            weights.dynamic_weight("0.5*mw+0.5*ew"),
            weights.optimal_weight(
                kirin_config=kirin_config,
                loss_type="MSE",
                max_weight=0.08,
                threshold_weight=0.05,
                bound_sum_threshold_weight=0.4,
                bound_gics={"sector": 0.5, "industry": 0.24},
                bound_financials_sector={"40": 0.048},
            ),
        ),
    factor=False,
    market_percentile=0.2,

    gics=False,
    gics_level=["sector"],
)

st.backtest()


# cp = Comparison(
#     data_dir=DATA_DIR,
#     identifier=IDENTIFIER,
#     kirin_config=kirin_config,
#     performance_fps=list(st.get_performance_fps()),
#     performance_names=list(st.get_performance_names()),
#     standard_benchmarks=["U:MTUM", "U:SPMO"],
#     comparison_periods=[],
#     final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
#     neptune_params=neptune_dict
# )
#
# cp.compare()
