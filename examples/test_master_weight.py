import sys


sys.path.append(".")

from paths import DATA_DIR, NEPTUNE_TOKEN

from utils.master_universe_gvkey import *


from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores


###################################################################################################
############################################# 설정부분 ##############################################
###################################################################################################

IDENTIFIER = "alt_ff_test"
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

DATE_FROM = "1999-12-31"
DATE_TO = "2021-04-30"
# DATE_TO = "2001-01-31"


neptune_dict = {
    "use_neptune": False,  #
    "user_id": "jayden",
    "project_name": "jayden",
    "exp_name": "",
    "description": "",
    "hparams": {},
    "tags": [],
    "exp_id": ["NEW"],
    "token": NEPTUNE_TOKEN,
}

###################################################################################################
###################################################################################################
###################################################################################################
setting_params = {
    "identifier": IDENTIFIER,  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "ff_master_cache").as_posix(),
        "use_sub_server": True,
        "exchange": ["NYSE", "NASDAQ", "OTC"],
        "security_type": ["COMMON", "ADR"],
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
        identifier=IDENTIFIER,
        kirin_config=setting_params['kirin_config'],

        date_from=DATE_FROM,
        date_to=DATE_TO,
        rebalancing_terms='M',

        long_picking_config=picks.component_picking(var_cf=0.5),
        long_weighting_config=(
            weights.market_tilting_weight(signal_coef=0.5),
            # weights.optimal_weight(
            #     kirin_config=setting_params['kirin_config'],
            #     loss_type="MSE",
            #     max_weight=0.08,
            #     threshold_weight=0.05,
            #     bound_sum_threshold_weight=0.4,
            #     bound_gics={"sector": 0.5, "industry": 0.24},
            #     bound_financials_sector={"40": 0.048},
            # ),
        ),

        factor=False,
        market_percentile=0.2,

        gics=False,
        gics_level=["sector"],

        infer_analysis_metrics = [
            ["Quantile_Return", 10, ["eq_weight", "v_weight"]],
            ["Quantile_Precision", 10],
            ["Linear_Regression"],
            ["Confusion_Matrix", ["above_zero", "above_median"], "S&PCOMP"]],
        infer_weight_analysis_metrics = [
            ["Quantile_Return", 10, ["eq_weight", "v_weight", "weight_file"]],
            ["Quantile_Precision", 10],
            ["Confusion_Matrix",
                ["weight_as_positive", "above_v_weight", "above_zero", "above_eq_weight",
                    "above_median"], 0.0],
            ["Linear_Regression"]]
    )

    st.backtest()


    cp = Comparison(
        data_dir=DATA_DIR,
        identifier=IDENTIFIER,
        kirin_config=setting_params['kirin_config'],
        performance_fps=list(st.get_performance_fps()),
        performance_names=list(st.get_performance_names()),
        standard_benchmarks=["U:MTUM", "U:SPMO"],
        comparison_periods=[],
        final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
        neptune_params=neptune_dict,
    )
    cp.compare()




