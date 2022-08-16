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
from utils.master_universe_gvkey import *
from utils.ff_manager import FFManager
from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from qraft_data.data import Tag as QraftDataTag
from strategy_integration.components.integration.deep.deep_integration import (
    DeepIntegration,
)
from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores
from strategy_simulation.strategy.weighting import _add_mw_tw_ew
from experiment.masters.ym_models import TestModel, TestModel2


###################################################################################################
############################################# 설정부분 ##############################################
###################################################################################################

IDENTIFIER = "after_ff_retry"
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

DATE_FROM = "1999-12-31"
DATE_TO = "2000-12-31"
# DATE_TO = "2001-01-31"

EPOCHS = 23
BATCH_SIZE = 32
CROSS_NUM_ITER = 100
CROSS_NUM_UNIVERSE = 256
TRAINING_LENGTH = 36
SAMPLE_LENGTH = 36


def test_picking(infer: pd.DataFrame):
    data = infer.values
    num = data.shape[1]

    def negative_opt(w, factors, factor_covariance, factor_stds):
        factor_returns = (factors * w).sum(axis=1)
        expected_return = factor_returns.sum()
        factor_exposures = factor_returns / factor_stds
        var = factor_exposures.dot(factor_exposures) #* (17/16) + 1e-10 # + factor_stds.mean()
#         var = factor_exposures.dot(factor_covariance).dot(factor_exposures) + 1e-6
        print(expected_return, np.sqrt(var))

        return -(expected_return - 0.*var)  # np.sqrt(var)

    def sum_is_one(x):
        return np.sum(x) - 1.0

    bounds = [(0.0, 1.0) for _ in range(num)]
    cons = {"type": "eq", "fun": sum_is_one}

    with np.errstate(divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cov = np.nan_to_num(np.corrcoef(data))
#             cov = np.nan_to_num(np.cov(data))

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

    return pd.Series(opts.x / np.sum(opts.x), index=infer.columns).sort_values(
        ascending=False
    )


def test_weight(dt, signal, kirin_config, mv, tv):
    df = _add_mw_tw_ew(dt, signal, kirin_config, mv, tv)
    w_sig = df["signal"]
    w_mv = df["mw"]

    cf = 0.15
    #     cf = 1.0
    w = cf * w_sig + (1 - cf) * w_mv
    w = w / w.sum()
    return w.sort_values(ascending=False)


strategy_dict = {
    "from_which": "infer",
    "long_amount": 1.0,
    "long_picking_config": {
        "name": "custom_picking",
        "custom_picking_func": test_picking,
    },
    "long_weighting_config": (
        {"name": "custom_weighting", "custom_weighting_func": test_weight},
    ),
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
    "project_name": "aiden",  # 프로젝트는 jayden, tei, aiden 으로 만들어 뒀습니다
    "exp_name": "ym_test",  # 실험의 이름 필수는 아닙니다
    "description": "ym try to attach ff",  # 실험의 설명 필수는 아닙니다
    "hparams": {
        **strategy_dict
    },  # 저장하고 싶은 하이퍼 파라미터를 딕셔너리로, 굳이 안넣어줘도 소스코드가 저장되어 당시의 셋팅을 확인가능합니다.
    "tags": [],  # 마스터 프로젝트에서는 태그를 변경하지 않습니다
    "exp_id": ["NEW"],  # 마스터 프로젝트에서는 EXP_ID를 변경하지 않습니다
    "token": NEPTUNE_TOKEN,  # 키는 고정입니다
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
universe_mapper = {
    "60988f94af024840cc4c2c8f": "all",  # All
    "60988f99af024840cc4c2cbc": "45,50",  # IT, Communication Services (84 stocks)
    "60988f97af024840cc4c2cad": "35",  # Health Care (21 stocks)
    "60988f95af024840cc4c2c9d": "25",  # Consumer Discretionary (12 stocks)
    "60988f98af024840cc4c2cb5": "40,60",  # Financials, Real Estate (4 stocks)
    "60988f94af024840cc4c2c95": "20",  # Industrials (3 stocks)
    "60988f93af024840cc4c2c7b": "10,55",  # Energy, Utilities (0 stocks)
    "60988f93af024840cc4c2c84": "15",  # Materials (0 stocks)
    "60988f96af024840cc4c2ca4": "30",  # Consumer Staples (0 stocks)
}

factorfactory_dict = {
    "universe_ids": list(universe_mapper.keys()),
    "metrics": [("sharpe", -1), ("sharpe", 1), ("hitratio", -1), ("mdd", -1)],
    "num_factors": 1,
    "window_length": 60,
    "num_buffer_factors": 5,
    #"description": "project_masters_20210430_mixweight",
    "description": None,
    "cache_dir": "/raid/sr-storage/ff_factors_20210430",  # DGX-v2
    # "cache_dir": "/raid/public/sr-storage/ff_factors_20210430",  # DGX-v1
}

# Preprocess input dataset
input_data = [
    ("pr_1m_0m", ["high_level", "equity", "get_monthly_price_return"], [1, 0]),
    ("mv", ["high_level", "equity", "get_monthly_market_value"]),
    ("btm", ["high_level", "equity", "get_book_to_market"]),
    ("mom_12m_1m", ["high_level", "equity", "get_monthly_price_return"], [12, 1]),
    ("ram_12m_0m", ["high_level", "equity", "get_ram"], ["pi", 12]),
    ("vol_3m", ["high_level", "equity", "get_monthly_volatility"], [3]),
    ("res_mom_12m_1m_0m", ["high_level", "equity", "get_res_mom"], [12, 1, 0]),
    ("res_vol_6m_3m_0m", ["high_level", "equity", "get_res_vol"], [6, 3, 0]),
    ("at", ["high_level", "equity", "get_asset_turnover"]),
    ("gpa", ["compustat", "custom_api", "get_gpa"]),
    ("rev_surp", ["high_level", "equity", "get_revenue_surprise"]),
    ("cash_at", ["high_level", "equity", "get_cash_to_asset"]),
    ("op_lev", ["high_level", "equity", "get_operating_leverage"]),
    ("roe", ["high_level", "equity", "get_roe"]),
    ("std_u_e", ["high_level", "equity", "get_standardized_unexpected_earnings"]),
    ("ret_noa", ["high_level", "equity", "get_return_on_net_operating_asset"]),
    ("etm", ["high_level", "equity", "get_earnings_to_market"]),
    (
        "ia_mv",
        ["high_level", "equity", "get_linear_assumed_intangible_asset_to_market_value"],
    ),
    ("ae_m", ["high_level", "equity", "get_advertising_expense_to_market"]),
    (
        "ia_ta",
        ["high_level", "equity", "get_linear_assumed_intangible_asset_to_total_asset"],
    ),
    ("rc_a", ["high_level", "equity", "get_rnd_capital_to_asset"]),
    ("r_s", ["high_level", "equity", "get_rnd_to_sale"]),
    ("r_a", ["high_level", "equity", "get_rnd_to_asset"]),
    ("fred_ff", ["fred", "ff"]),
    ("t3m", ["high_level", "macro", "get_us_treasury_3m"]),
    ("t6m", ["fred", "treasury_6m"]),
    ("t2y", ["high_level", "macro", "get_us_treasury_2y"]),
    ("t3y", ["fred", "treasury_3y"]),
    ("t5y", ["high_level", "macro", "get_us_treasury_5y"]),
    ("t7y", ["fred", "treasury_7y"]),
    ("t10y", ["high_level", "macro", "get_us_treasury_10y"]),
    ("aaa", ["fred", "aaa_corporate"]),
    ("baa", ["fred", "baa_corporate"]),
    ("snp500_pr", ["high_level", "index", "get_snp500_momentum"], [], {"periods": 1}),
    (
        "wilshire500_pr",
        ["high_level", "index", "get_wilshire5000_price_index_momentum"],
        [],
        {"periods": 1},
    ),
    ("ted", ["fred", "ted_spread"]),
    ("t1y_ff", ["fred", "t1y_minus_ff"]),
    ("t5y_ff", ["fred", "t5y_minus_ff"]),
    ("t10y_t2y", ["fred", "t10y_minus_t2y"]),
    ("aaa_t10y", ["fred", "aaa_minus_t10y"]),
    ("baa_t10y", ["fred", "baa_minus_t10y"]),
    ("aaa_ff", ["fred", "aaa_minus_ff"]),
    ("baa_ff", ["fred", "baa_minus_ff"]),
    ("core_cpi", ["high_level", "macro", "get_core_cpi_rate"]),
    ("core_pce", ["high_level", "macro", "get_core_pce_rate"]),
    ("core_ppi", ["high_level", "macro", "get_core_ppi_rate"]),
    ("cpi", ["high_level", "macro", "get_cpi_rate"]),
    ("pce", ["high_level", "macro", "get_pce_rate"]),
    ("ppi", ["high_level", "macro", "get_ppi_rate"]),
    (
        "trimmed_pce",
        ["high_level", "macro", "get_trimmed_mean_pce_rate"],
        [],
        {"change_to": "none"},
    ),
    ("unemploy", ["high_level", "macro", "get_unemployment_rate"]),
    ("retail_mfr", ["high_level", "macro", "get_retail_money_funds_rate"]),
    ("m1", ["high_level", "macro", "get_us_m1_rate"]),
    ("m2", ["high_level", "macro", "get_us_m2_rate"]),
    ("export_growth", ["fred", "exports_growth"]),
    ("import_growth", ["fred", "imports_growth"]),
    ("real_gig", ["fred", "real_government_investment_growth"]),
    ("real_pig", ["fred", "real_private_investment_growth"]),
    (
        "federal_tg",
        ["high_level", "macro", "get_federal_government_current_tax_receipts_growth"],
    ),
    ("real_gdp", ["fred", "real_gdp_growth"]),
    ("corporate_tg", ["high_level", "macro", "get_corporate_profits_after_tax_growth"]),
    (
        "industrial_prod",
        ["high_level", "macro", "get_industrial_production_index_rate"],
    ),
    ("home_pr", ["high_level", "macro", "get_home_price_index_rate"]),
    ("wti", ["high_level", "macro", "get_wti_price_rate"]),
    ("capa_util", ["fred", "capacity_utilization"]),
    ("snp500_pe", ["high_level", "macro", "get_snp_pe"]),
    ("snp500_vol", ["high_level", "macro", "get_snp500_vol"]),
]

integrated_model_params = {
    "training_all_at_once": False,
    "reinitialization": 1,
    "models": [
        {
            "name": "custom_model",
            "custom_model": TestModel,
            "inputs": ["x", "ffs"],
            "targets": ["y"],
            "forwards": [],
            "cross_units": [64, 32],
            "ff_cross_units": [8, 4],
            "batch_units": [32, 16],
            "dropout": 0.0,
            "layer_norm": True,
            "noise": 0.01,
            "learning_rate": 3e-4,
            "weight_decay": 3e-5,
            "huber_loss_positive_beta": 1.0,
            "huber_loss_negative_beta": 1.0,
            "stage": (10, 20),
        },
    ],
    "forwards_at_epoch": [],
    "forward_names": [],
    "early_stopping_start_after": 100,
    "early_stopping_interval": 10,
    "epochs": EPOCHS,
    "path": DATA_DIR,
    "save_all_epochs": False,
    "weight_share": False,
    "batch_update": ["optimizer"],
    "epoch_update": [],
    "batch_size": BATCH_SIZE,
    "validation_ratio": 0.4,
    "shuffle": True,
}


def run_model(_, context_name_list, context_list, dict_of_shapes, inference_date):
    di = DeepIntegration(
        identifier=setting_params["identifier"],
        sub_identifier=inference_date.strftime("%Y-%m-%d"),
        dataset=None,
        data_shape=dict_of_shapes,
        integrated_model_params=integrated_model_params,
        setting_params=setting_params,
    )
    training_lc = context_list[0]
    inference_lc_list = context_list

    split_item: bm.splitting.TrainingValidationSplit = training_lc.training_split(
        bm.splitting.training_validation_split,
        validation_ratio=integrated_model_params["validation_ratio"],
        shuffle=integrated_model_params["shuffle"],
    )

    early_stopping, best_loss = di.initialize_train()
    early_stopping_flag = False
    for epoch in range(1, integrated_model_params["epochs"] + 1):
        if early_stopping_flag:
            break
        train_step_info_list = []
        val_step_info_list = []
        train_stack_forwards = {}
        val_stack_forwards = {}
        training_flags = True

        di._set_all_train_mode()
        for meta, data in training_lc.split_of(split_item.TRAINING).training_iterate(
            batch_size=integrated_model_params["batch_size"],
            inclusive=True,
            same_time=False,
            probabilistic_sampling=True,
            drop_last_if_not_probabilistic_sampling=True,
            cross_sampling=True,
            cross_num_iter=CROSS_NUM_ITER,
            cross_num_element=CROSS_NUM_UNIVERSE,
        ):
            step_info = di.train(data)
            train_step_info_list.append(step_info)

        best_loss, early_stopping = di.summary_of_results(
            epoch,
            train_step_info_list,
            best_loss,
            train_stack_forwards,
            early_stopping,
            training_flags,
        )
        print(
            f'training: {epoch}/{integrated_model_params["epochs"]} ({best_loss[0]:.4f})'
        )

        training_flags = False
        di._set_all_validation_mode()
        for meta, data in training_lc.split_of(split_item.VALIDATION).training_iterate(
            batch_size=integrated_model_params["batch_size"],
            inclusive=True,
            same_time=False,
            probabilistic_sampling=True,
            drop_last_if_not_probabilistic_sampling=True,
            cross_sampling=True,
            cross_num_iter=CROSS_NUM_ITER,
            cross_num_element=CROSS_NUM_UNIVERSE,
        ):
            step_info = di.validation(data)
            val_step_info_list.append(step_info)

        best_loss, early_stopping = di.summary_of_results(
            epoch,
            val_step_info_list,
            best_loss,
            val_stack_forwards,
            early_stopping,
            training_flags,
        )
        print(
            f'validation : {epoch}/{integrated_model_params["epochs"]} ({best_loss[0]:.4f})'
        )

        if early_stopping.is_stopping(epoch):
            print(f"early stopping at {epoch} with loss {best_loss[0]:.4f}")
            early_stopping_flag = True

        di._set_all_validation_mode()
        for i, (context_name, lc) in enumerate(
            zip(context_name_list, inference_lc_list)
        ):
            for meta, data in lc.inference_iterate(cross_sampling=True):
                sec_keys = meta["SECURITY"][0]

                if integrated_model_params["save_all_epochs"]:
                    di.infer_with_all_epochs(None, data, sec_keys)

                if epoch == integrated_model_params["epochs"] or early_stopping_flag:
                    if i == 0:
                        di.infer_with_all_epochs(None, data, sec_keys)
                    else:
                        di.infer_with_all_epochs(f"{context_name}", data, sec_keys)
                    di.summary_after_infer(
                        epoch, integrated_model_params["epochs"], best_loss
                    )
        di.run_after_epoch()


if __name__ == "__main__":

    ffmanager = FFManager(**date_dict, **factorfactory_dict)
    ff_columns = ffmanager.columns

    universe = Universe(**setting_params["kirin_config"])
    api = get_kirin_api(universe)
    load_dataset = LoadData(path=Path(DATA_DIR) / "ff_master_dataset", universe=universe)

    with mp.Pool(8) as pool:
        raw_inputs = pool.starmap(load_dataset.call_if_not_loaded, input_data)

    sector_values = load_dataset.call_if_not_loaded(
        "sector_values",
        ["compustat", "read_sql"],
        ["SELECT giccd FROM r_giccd WHERE gictype='GSECTOR';"],
    )
    sector_values = sector_values.values.reshape(-1)

    inputs = []
    for raw_data in raw_inputs:
        if raw_data.get_tag() == QraftDataTag.INDEX.value:
            raw_data = raw_data.rolling(SAMPLE_LENGTH).zscore()
        if raw_data.get_tag() == QraftDataTag.EQUITY.value:
            if raw_data.name == "sector":
                raw_data._data = raw_data._data.fillna(-1).astype(int).astype(str)
                raw_data = raw_data.slice(columns=ff_columns).one_hot(sector_values)
            else:
                raw_data = 2 * raw_data.slice(columns=ff_columns).rank(pct=True) - 1

        inputs.append(raw_data)

    # preprocess mv & output
    y = load_dataset.call_if_not_loaded(
        "tr_1m_0m", ["high_level", "equity", "get_monthly_price_return"], [1, 0]
    )
    y = 2 * y.slice(columns=ff_columns).rank(pct=True) - 1

    # training filter
    training_filter = load_dataset.call_if_not_loaded(
        "price", ["high_level", "equity", "get_monthly_price_data"]
    )
    training_filter._data = training_filter._data.notna()
    training_filter = training_filter.slice(columns=ff_columns)

    # inference filter
    infer_filter_base = training_filter.copy()
    infer_filter_base = infer_filter_base.slice(columns=ff_columns)

    probability = training_filter.copy()
    probability._data = probability._data.astype(float)
    probability = probability.slice(columns=ff_columns)

    infer_universes = [
        UNIVERSE1,
        UNIVERSE2,
        UNIVERSE3,
        UNIVERSE19951231,
        UNIVERSE20021231,
        UNIVERSE20101231,
        UNIVERSE20151231,
    ]
    name_list = ["U1", "U2", "U3", "SU95", "SU02", "SU10", "SU15"]

    infer_filter_list = []
    for universe_gk in infer_universes:
        infer_filter = infer_filter_base.copy()
        difference = infer_filter._data.columns.difference(universe_gk)
        infer_filter.loc[:, difference] = False
        infer_filter_list.append(infer_filter)

    # Check input, output and filter dataset has same index and columns
    index, columns = bm.checking.check_equal_index_and_columns(
        inputs + [y, infer_filter_base]
    )

    sbm_list = []
    for name, infer_filter in zip(name_list, infer_filter_list):
        binder_x = bm.DataBinderV3(
            data_list=inputs,
            training_filter=training_filter,
            inference_filter=infer_filter,
            length=SAMPLE_LENGTH,
            is_input=True,
            max_nan_ratio=1.0,
            aug=None,
        )
        binder_y = bm.DataBinderV3(
            data_list=[y],
            training_filter=training_filter,
            inference_filter=infer_filter,
            length=1,
            is_input=False,
            max_nan_ratio=0.0,
            aug=None,
        )
        sbm = bm.SecurityBatchMaker(
            save_path=(
                Path(DATA_DIR) / setting_params["identifier"] / "batch_dataset" / name
            ).as_posix(),
            index=index,
            columns=columns,
            data_map={"x": binder_x, "y": binder_y},
            max_cache="30GB",
            probability=probability,
        )
        sbm_list.append(sbm)

    dates_inference = pd.date_range(
        start=date_dict["date_from"],
        end=date_dict["date_to"],
        freq=date_dict["rebalancing_terms"],
    ).to_pydatetime()
    
 
    
#     dates_inference = [
#          datetime.datetime(2002, 2, 28),
#          datetime.datetime(2002, 12, 31),
#          datetime.datetime(2003, 1, 31),
#          datetime.datetime(2006, 7, 31),
#          datetime.datetime(2010, 5, 31),
#          datetime.datetime(2013, 1, 31),
#          datetime.datetime(2020, 5, 31),
#     ]
    shapes_dict = sbm_list[0].get_sample_shapes()

    run_proc = MultiProcess()
    run_proc.cpu_count(max_count=setting_params["cpu_count"])

    for d_inference in dates_inference:
        length_list = [TRAINING_LENGTH] + [0] * (len(sbm_list) - 1)

        ff_datas = ffmanager.datas_on_date(d_inference)
        ff_datas = [data.slice(columns=columns) for data in ff_datas]
        ff_datas = [2*data.rank(pct=True) - 1 for data in ff_datas]

        input_ffs = bm.DataBinderV3(
            data_list=ff_datas,
            training_filter=training_filter,
            inference_filter=training_filter,
            length=SAMPLE_LENGTH,
            is_input=True,
            max_nan_ratio=1.0,
            aug=None,
        )
        sbm_ff = bm.SecurityBatchMaker(
            save_path=(
                Path(DATA_DIR) / setting_params["identifier"] / "batch_dataset" / "ffs"
            ).as_posix(),
            index=index,
            columns=columns,
            data_map={"ffs": input_ffs},
            max_cache="30GB",
            probability=None,
        )
        lc_list = []
        shapes_dict.update(sbm_ff.get_sample_shapes())

        for sbm, length in zip(sbm_list, length_list):
            lc_origin = sbm.local_context(inference_dates=d_inference, length=length)
            lc_ff = sbm_ff.local_context(inference_dates=d_inference, length=length)
            lc = lc_origin.merge(lc_ff)
            lc_list.append(lc)

#** 디버깅용
#       run_model("_", name_list, lc_list, shapes_dict, d_inference)

        proc = mp.spawn(run_model, args=(name_list, lc_list, shapes_dict, d_inference), join=False).processes[0]
        run_proc.run_process(proc)

    run_proc.final_process()

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
        "UNIVERSE1_MCAP",
        "UNIVERSE1_DOLLAR",
        "UNIVERSE20021231_MCAP",
        "UNIVERSE20021231_DOLLAR",
        "UNIVERSE20101231_MCAP",
        "UNIVERSE20101231_DOLLAR",
        "UNIVERSE20151231_MCAP",
        "UNIVERSE20151231_DOLLAR",
        "UNIVERSE19951231_MCAP",
        "UNIVERSE19951231_DOLLAR",
        "UNIVERSE2_MCAP",
        "UNIVERSE2_DOLLAR",
        "UNIVERSE3_MCAP",
        "UNIVERSE3_DOLLAR",
    ]

    INFER_LIST = ["U1", "SU02", "SU10", "SU15", "SU95", "U2", "U3"]

    for i in range(len(INFER_LIST)):
        comparison_dict["performance_fps"] = [p_pfs_list[i]]
        comparison_dict["performance_names"] = [p_names_list[i]]
        comparison_dict["standard_benchmarks"] = UNIVERSE_LIST[2 * i : 2 * (i + 1)]

        neptune_dict["tags"] = [NEPTUNE_IDENTIFIER] + [INFER_LIST[i]]

        cp = Comparison(
            data_dir=DATA_DIR,
            setting_params=setting_params,
            neptune_params=neptune_dict,
            **comparison_dict,
        )
        cp.compare()
