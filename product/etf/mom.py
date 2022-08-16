import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


from qraft_data.data import QraftData
from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from pathlib import Path
from utils.load_data import LoadData

from pathlib import Path
from utils.load_data import LoadData


MD_PATH = Path('/Users/jaehoon/QRAFT/git_clones/tem_etf_transfer/compustat_ergate/monthly_data')
AMOM =  ( MD_PATH / 'AMOM' )

setting_params = {
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON"],
        "backtest_mode": True,
        "except_no_isin_code": False,
        "class_a_only": True,
        "pretend_monthend": False,
    },
}


input_data = [
    ("mom_1m_0m", ["high_level", "equity", "get_mom"], ["pi", 1, 0]),
    ("mom_2m_0m", ["high_level", "equity", "get_mom"], ["pi", 2, 0]),
    ("mom_3m_0m", ["high_level", "equity", "get_mom"], ["pi", 3, 0]),
    ("mom_4m_0m", ["high_level", "equity", "get_mom"], ["pi", 4, 0]),
    ("mom_5m_0m", ["high_level", "equity", "get_mom"], ["pi", 5, 0]),
    ("mom_6m_0m", ["high_level", "equity", "get_mom"], ["pi", 6, 0]),
    ("mom_7m_0m", ["high_level", "equity", "get_mom"], ["pi", 7, 0]),
    ("mom_8m_0m", ["high_level", "equity", "get_mom"], ["pi", 8, 0]),
    ("mom_9m_0m", ["high_level", "equity", "get_mom"], ["pi", 9, 0]),
    ("mom_10m_0m", ["high_level", "equity", "get_mom"], ["pi", 10, 0]),
    ("mom_11m_0m", ["high_level", "equity", "get_mom"], ["pi", 11, 0]),
    ("mom_12m_0m", ["high_level", "equity", "get_mom"], ["pi", 12, 0]),
]

universe = Universe(**setting_params["kirin_config"])
api = get_kirin_api(universe)
load_dataset = LoadData(path= Path(DATA_DIR) / "kirin_dataset", universe=universe)

import multiprocessing as mp
with mp.Pool(10) as pool:
    res = pool.starmap(load_dataset.call_if_not_loaded, input_data)

for i, qdata in enumerate(res):
    qdata = qdata.winsorize((0.01, 0.99), pct=True).zscore()
    input_data[i] = qdata