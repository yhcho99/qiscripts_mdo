import sys
import warnings

sys.path.append("")

import pandas as pd
import numpy as np
import scipy.optimize
import batch_maker as bm
import torch.multiprocessing as mp

from pathlib import Path
from paths import DATA_DIR, NEPTUNE_TOKEN
from utils.load_data import LoadData
from utils.master_universe_gvkey import *
from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from qraft_data.data import Tag as QraftDataTag
from strategy_integration.components.integration.deep.deep_integration import DeepIntegration
from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores
from strategy_simulation.strategy.weighting import _add_mw_tw_ew
from experiment.masters.th_models import TestModel
import time

import sys

risk_cf = int(sys.argv[1])

IDENTIFIER = f"test_{risk_cf:04}"
ACTIVE_CF = 0.5
RISK_CF = 0.1 * risk_cf
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

DATE_FROM = "1995-12-31"
DATE_TO = "2021-05-31"

EPOCHS = 22
BATCH_SIZE = 32
CROSS_NUM_ITER = 100
CROSS_NUM_UNIVERSE = 256
TRAINING_LENGTH = 6
SAMPLE_LENGTH = 36


def test_picking(infer: pd.DataFrame, ddd):
    print(ddd)

    data = infer.values
    with np.errstate(divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.nan_to_num(np.cov(data))

    with np.errstate(divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stds = np.nan_to_num(np.std(data, axis=1))

    # ss = infer.std(1)
    # col = (infer.sum(0) / infer.div(ss, axis=0).pow(2).sum(0)).sort_values(ascending=False).iloc[:350].index
    # infer = infer.loc[infer.std(1).sort_values().iloc[[0,1,-1,-2]].index]
    col = infer.sum(axis=0).sort_values(ascending=False).iloc[:350].index
    # col = col.union(infer[infer > 0].sum(axis=0).sort_values(ascending=False).iloc[:25].index)
    # col = col.append(infer.sum(axis=0).sort_values(ascending=False).iloc[100:110].index)
    # col = col.append(infer.sum(axis=0).sort_values(ascending=False).iloc[150:160].index)
    # col = col.append(infer.sum(axis=0).sort_values(ascending=False).iloc[200:210].index)
    infer = infer[col]

    data = infer.values
    num = data.shape[1]

    def negative_opt(w, factors, factor_stds, corr):
        global RISK_CF

        factor_returns = (factors * w).sum(axis=1)
        expected_return = factor_returns.sum()
        factor_exposures = factor_returns / factor_stds
        # var = factor_exposures.dot(corr).dot(factor_exposures) + 1e-10
        var = factor_exposures.dot(factor_exposures) + 1e-10
        # print(expected_return, np.sqrt(var))
        print(expected_return, var)
        return -(expected_return - RISK_CF*var)
        # return -expected_return  # /np.sqrt(var)

    def sum_is_one(x):
        return np.sum(x) - 1.0

    bounds = [(0.1/num, 1.0) for _ in range(num)]
    cons = {"type": "eq", "fun": sum_is_one}

    # N = 30
    # ww = [0]
    # for i in range(N, 0, -1):
    #     ww.append(ww[-1] + i/N)
    # ww = ww[::-1][:-1]
    # ss = sum(ww)
    # ww = [e/ss for e in ww]
    # ww = [1/N] * N
    #
    # weight = ww + [0] * (num - N)
    weight = np.full(num, 1 / num, dtype=float)

    opts = scipy.optimize.minimize(
        negative_opt,
        weight,
        args=(data, stds, corr),
        method="SLSQP",
        bounds=bounds,
        constraints=cons
    )
    if not opts.success:
        warnings.warn(f"optimal weight not optimized.")
    weight = opts.x

    s = pd.Series(weight, index=infer.columns).sort_values(ascending=False)
    s = s / s.sum()

    return s


def test_weight(dt, signal, kirin_config, mv, tv):
    global ACTIVE_CF

    print("weight", dt)

    df = _add_mw_tw_ew(dt, signal, kirin_config, mv, tv)
    w_sig = df["signal"]
    w_mv = df["mw"]

    w = ACTIVE_CF * w_sig + (1 - ACTIVE_CF) * w_mv
    w = w.sort_values(ascending=False).iloc[:350]
    w = w / w.sum()
    return w


strategy_dict = {
    "from_which": "infer",
    "long_amount": 1.0,
    "long_picking_config": {"name": "custom_picking", "custom_picking_func": test_picking},
    "long_weighting_config": ({"name": "custom_weighting", "custom_weighting_func": test_weight},),
    "short": False,
    "short_amount": 0.0,
    "short_picking_config": picks.picking_by_signal("out", False, 1, None, ascending=True),
    "short_weighting_config": weights.market_weight(),
    "weight_adjusting_unitdate": False,
    "backtest_daily_out": False,
    "backtest_daily_out_lag": [0, 1],
    "factor": False,
    "save_factor": False,
    "market_percentile": 0.2,
    "gics": True,
    "save_gics": True,
    "gics_level": ["sector"],
}
comparison_dict = {
    "performance_fps": [],
    "performance_names": [],
    "standard_benchmarks": ["S&PCOMP", "NASA100"],
    "comparison_periods": [
        ("2000-12-31", "2021-05-31"),
        ("2008-12-31", "2021-05-31"),
        ("2012-12-31", "2017-12-31"),
        ("2012-12-31", "2021-05-31"),
        ("2018-12-31", "2021-05-31"),
    ],
    "final_score": final_scores.annualized_return(exponential_decay_rate=None, total=False),
}
neptune_dict = {
    "use_neptune": False,  # ????????? ??????????????? True ??? ???????????????.
    "user_id": "jayden",  # USER_ID??? ????????? ?????? ???????????????
    "project_name": "tei",  # ??????????????? jayden, tei, aiden ?????? ????????? ????????????
    "exp_name": "test_test",  # ????????? ?????? ????????? ????????????
    "description": "th various try",  # ????????? ?????? ????????? ????????????
    "hparams": {**strategy_dict},  # ???????????? ?????? ????????? ??????????????? ???????????????, ?????? ??????????????? ??????????????? ???????????? ????????? ????????? ?????????????????????.
    "tags": [],  # ????????? ????????????????????? ????????? ???????????? ????????????
    "exp_id": ['NEW'],  # ????????? ????????????????????? EXP_ID??? ???????????? ????????????
    "token": NEPTUNE_TOKEN,
}

setting_params = {
    "identifier": IDENTIFIER,  # ????????? ???????????????
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON"],
        "backtest_mode": True,
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
    "tensorboard_summary_save": False,
    "cpu_count": 12
}

date_dict = {
    "date_from": DATE_FROM,
    "date_to": DATE_TO,
    "rebalancing_terms": "M",
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
    ("ia_mv", ["high_level", "equity", "get_linear_assumed_intangible_asset_to_market_value"]),
    ("ae_m", ["high_level", "equity", "get_advertising_expense_to_market"]),
    ("ia_ta", ["high_level", "equity", "get_linear_assumed_intangible_asset_to_total_asset"]),
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
    ("wilshire500_pr", ["high_level", "index", "get_wilshire5000_price_index_momentum"], [], {"periods": 1}),
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
    ("trimmed_pce", ["high_level", "macro", "get_trimmed_mean_pce_rate"], [], {"change_to": "none"},),
    ("unemploy", ["high_level", "macro", "get_unemployment_rate"]),
    ("retail_mfr", ["high_level", "macro", "get_retail_money_funds_rate"]),
    ("m1", ["high_level", "macro", "get_us_m1_rate"]),
    ("m2", ["high_level", "macro", "get_us_m2_rate"]),
    ("export_growth", ["fred", "exports_growth"]),
    ("import_growth", ["fred", "imports_growth"]),
    ("real_gig", ["fred", "real_government_investment_growth"]),
    ("real_pig", ["fred", "real_private_investment_growth"]),
    ("federal_tg", ["high_level", "macro", "get_federal_government_current_tax_receipts_growth"]),
    ("real_gdp", ["fred", "real_gdp_growth"]),
    ("corporate_tg", ["high_level", "macro", "get_corporate_profits_after_tax_growth"]),
    ("industrial_prod", ["high_level", "macro", "get_industrial_production_index_rate"]),
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
            "inputs": ["x"],
            "targets": ["y"],
            "forwards": [],
            "cross_units": [64, 32],
            "batch_units": [32, 16],
            "dropout": 0.0,
            "layer_norm": True,
            "noise": 0.01,
            "learning_rate": 3e-4,
            "weight_decay": 3e-5,
            "huber_loss_positive_beta": 1.0,
            "huber_loss_negative_beta": 1.0,
            "stage": (10, 20)
        },
    ],
    "forwards_at_epoch": [],
    "forward_names": [],
    "early_stopping_start_after": 100,
    "early_stopping_interval": 10,
    "adversarial_info": {
        "adversarial_use": False,
        "adversarial_alpha": 0.0,
        "adversarial_weight": 0.0,
    },
    "epochs": EPOCHS,
    "path": DATA_DIR,
    "save_all_epochs": False,
    "weight_share": False,
    "batch_update": ["optimizer"],
    "epoch_update": [],
    "validation_ratio": 0.4,
    "shuffle": True,
}


def run_model(_, local_context, dict_of_shapes, inference_date):
    di = DeepIntegration(
        identifier=setting_params["identifier"],
        sub_identifier=inference_date.strftime("%Y-%m-%d"),
        dataset=local_context,
        data_shape=dict_of_shapes,
        integrated_model_params=integrated_model_params,
        setting_params=setting_params,
    )
    split_item: bm.splitting.TrainingValidationSplit = local_context.training_split(
        bm.splitting.training_validation_split,
        validation_ratio=integrated_model_params["validation_ratio"],
        shuffle=integrated_model_params["shuffle"],
    )

    early_stopping, best_loss = di.initialize_train()
    early_stopping_flag = False

    for epoch in range(1, EPOCHS + 1):
        if early_stopping_flag:
            break
        train_step_info_list = []
        val_step_info_list = []
        train_stack_forwards = {}
        val_stack_forwards = {}
        training_flags = True

        di._set_all_train_mode()
        for meta, data in local_context.split_of(split_item.TRAINING).training_iterate(
            batch_size=BATCH_SIZE,
            inclusive=True,
            same_time=False,
            probabilistic_sampling=True,
            drop_last_if_not_probabilistic_sampling=True,
            cross_sampling=True,
            cross_num_iter=CROSS_NUM_ITER,
            cross_num_element=CROSS_NUM_UNIVERSE
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
        print(f'training: {epoch}/{integrated_model_params["epochs"]} ({best_loss[0]:.4f})')

        training_flags = False
        di._set_all_validation_mode()
        for meta, data in local_context.split_of(split_item.VALIDATION).training_iterate(
            batch_size=BATCH_SIZE,
            inclusive=True,
            same_time=False,
            probabilistic_sampling=True,
            drop_last_if_not_probabilistic_sampling=True,
            cross_sampling=True,
            cross_num_iter=CROSS_NUM_ITER,
            cross_num_element=CROSS_NUM_UNIVERSE
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
        print(f'validation : {epoch}/{integrated_model_params["epochs"]} ({best_loss[0]:.4f})')

        if early_stopping.is_stopping(epoch):
            print(f"early stopping at {epoch} with loss {best_loss[0]:.4f}")
            early_stopping_flag = True

        di._set_all_validation_mode()
        for meta, data in local_context.inference_iterate(cross_sampling=True):
            sec_keys = meta["SECURITY"][0]

            if integrated_model_params["save_all_epochs"]:
                di.infer_with_all_epochs(None, data, sec_keys)

            if epoch == integrated_model_params["epochs"] or early_stopping_flag:
                di.infer_with_all_epochs(None, data, sec_keys)
                di.summary_after_infer(epoch, integrated_model_params["epochs"], best_loss)
        di.run_after_epoch()


if __name__ == "__main__":
    # universe = Universe(**setting_params["kirin_config"])
    # api = get_kirin_api(universe)
    # load_dataset = LoadData(path=Path(DATA_DIR) / "dataset", universe=universe)
    #
    # raw_inputs = [load_dataset.call_if_not_loaded(*e) for e in input_data]
    #
    # sector_values = load_dataset.call_if_not_loaded(
    #     "sector_values",
    #     ["compustat", "read_sql"],
    #     ["SELECT giccd FROM r_giccd WHERE gictype='GSECTOR';"]
    # )
    # sector_values = sector_values.values.reshape(-1)
    #
    # inputs = []
    # for raw_data in raw_inputs:
    #     if raw_data.get_tag() == QraftDataTag.INDEX.value:
    #         raw_data = raw_data.rolling(SAMPLE_LENGTH).zscore()
    #     if raw_data.get_tag() == QraftDataTag.EQUITY.value:
    #         if raw_data.name == "sector":
    #             raw_data = raw_data.one_hot(sector_values)
    #         else:
    #             raw_data = 2 * raw_data.rank(pct=True) - 1
    #
    #     inputs.append(raw_data)
    #
    # # preprocess mv & output
    # y = load_dataset.call_if_not_loaded("tr_1m_0m", ["high_level", "equity", "get_monthly_total_return"], [1, 0])
    # y = 2 * y.rank(pct=True) - 1
    #
    # # training filter
    # training_filter = load_dataset.call_if_not_loaded("price", ["high_level", "equity", "get_monthly_price_data"])
    # training_filter._data = training_filter._data.notna()
    #
    # # inference filter
    # inference_filter = load_dataset.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    # inference_filter = inference_filter.rank(ascending=False, pct=True) <= 0.15
    #
    # probability = training_filter.copy()
    # probability._data = probability._data.astype(float)
    #
    # # Check input, output and filter dataset has same index and columns
    # index, columns = bm.checking.check_equal_index_and_columns(
    #     inputs + [y, training_filter, inference_filter]
    # )
    # binder_x = bm.DataBinderV3(
    #     data_list=inputs,
    #     training_filter=training_filter,
    #     inference_filter=inference_filter,
    #     length=SAMPLE_LENGTH,
    #     is_input=True,
    #     max_nan_ratio=1.0,
    #     aug=None,
    # )
    # binder_y = bm.DataBinderV3(
    #     data_list=[y],
    #     training_filter=training_filter,
    #     inference_filter=inference_filter,
    #     length=1,
    #     is_input=False,
    #     max_nan_ratio=0.0,
    #     aug=None,
    # )
    # sbm = bm.SecurityBatchMaker(
    #     save_path=(Path(DATA_DIR) / setting_params["identifier"] / "batch_dataset").as_posix(),
    #     index=index,
    #     columns=columns,
    #     data_map={
    #         "x": binder_x,
    #         "y": binder_y,
    #     },
    #     max_cache="15GB",
    #     probability=probability,
    # )
    # dates_inference = pd.date_range(
    #     start=date_dict["date_from"],
    #     end=date_dict["date_to"],
    #     freq=date_dict["rebalancing_terms"],
    # ).to_pydatetime()
    #
    # shapes_dict = sbm.get_sample_shapes()
    #
    # for d_inference in dates_inference:
    #     lc = sbm.local_context(inference_dates=d_inference, length=TRAINING_LENGTH)
    #     run_model("_", lc, shapes_dict, d_inference)

    st = Strategy(
        data_dir=DATA_DIR,
        setting_params=setting_params,
        date_params=date_dict,
        **strategy_dict,
    )
    st.backtest()

    comparison_dict["performance_fps"] += st.get_performance_fps()
    comparison_dict["performance_names"] += st.get_performance_names()

    cp = Comparison(
        data_dir=DATA_DIR,
        setting_params=setting_params,
        neptune_params=neptune_dict,
        **comparison_dict,
    )
    cp.compare()
