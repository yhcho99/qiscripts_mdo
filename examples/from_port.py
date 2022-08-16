import sys
sys.path.append(".")
from pathlib import Path


from strategy_simulation.portfolio import Portfolio
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import final_scores
from paths import DATA_DIR

NEPTUNE_TOKEN = None

IDENTIFIER = 'amom_ft_final'
BENCHMARK = ["S&PCOMP"]

# IDENTIFIER = 'SIMULATION_QRFT'
# BENCHMARK = ["S&PCOMP", "NASA100"]
#"S&PCOMP"
#"U:MTUM"uyhuhhhhhhhhhuhhuyyyyyuuuyuyu
#"NASA100"

kirin_config = {
    "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
    "use_sub_server": False,
    "exchange": ["NYSE", "NASDAQ"],
    "security_type": ["COMMON"],
    "backtest_mode": True,
    "except_no_isin_code": False,
    "class_a_only": True,
    "primary_issue": False,
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

pf = Portfolio(
    kirin_config=kirin_config,
    data_dir=DATA_DIR,
    identifier=IDENTIFIER,

    date_from='2019-06-30',
    date_to='2020-08-31',
    rebalancing_terms='M',

    # backtest_daily_out=False,
    # backtest_daily_out_lag=[2],

    factor=True,

    gics=True,
    gics_level=['sector']
)

pf.backtest()

cp = Comparison(
    data_dir=DATA_DIR,
    identifier=IDENTIFIER,
    kirin_config=kirin_config,
    performance_fps=list(pf.get_performance_fps()),
    performance_names=list(pf.get_performance_names()),
    standard_benchmarks=BENCHMARK,
#    comparison_periods=[('2017-12-31','2019-12-31')],
    comparison_periods=[],
    final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
    neptune_params=neptune_dict,
)
cp.compare()









