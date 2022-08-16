from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from paths import DATA_DIR, NEPTUNE_TOKEN
from pathlib import Path

setting_params = {
    "identifier": "nvq_0531",  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON"],
        "backtest_mode": False,
        "except_no_isin_code": False,
        "class_a_only": True,
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

    "cpu_count": 24
}

universe = Universe(**setting_params["kirin_config"])
api = get_kirin_api(universe)


filter_mv = api.compustat.get_monthly_market_value()
index_book_to_market = api.high_level.equity.get_index_for_book_to_market()
index_linear_intangible_to_tangible = api.high_level.equity.index_for_linear_assumed_intangible_asset_to_total_asset()

index_book_to_market = index_book_to_market[filter_mv].rank(ascending=True, pct=True)
index_linear_intangible_to_tangible = index_linear_intangible_to_tangible[filter_mv].rank(ascending=True,pct=True)
v = 0.7 * index_book_to_market + 0.3 * index_linear_intangible_to_tangible
filter_infer = v.rank(ascending=False, pct=False) <= 300

