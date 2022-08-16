# 성과가 저장되어 있을때

import sys
sys.path.append(".")
from pathlib import Path

from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import final_scores

#데이터가 저장되는 경로 설정을 해줍니다
DATA_DIR = '/Users/jaehoon/QRAFT/data'
NEPTUNE_TOKEN = None

IDENTIFIER = 'SIMULATION_AMOM'
BENCHMARK = ["U:SPMO" , "U:MTUM"]


kirin_config = {
    "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
    "use_sub_server": False,
    "exchange": ["NYSE", "NASDAQ"],
    "security_type": ["COMMON"],
    "backtest_mode": True,
    "except_no_isin_code": False,
    "class_a_only": True,
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

cp = Comparison(
    data_dir=DATA_DIR,
    identifier=IDENTIFIER,
    kirin_config=kirin_config,
    performance_fps=[IDENTIFIER],
    performance_names=[IDENTIFIER],
    standard_benchmarks=BENCHMARK,
    comparison_periods=[],
    final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
    neptune_params=neptune_dict,
)
cp.compare()









