import sys
import time
sys.path.append("")
import pandas as pd
import batch_maker as bm
import multiprocessing as mp

from pathlib import Path
from paths import DATA_DIR, NEPTUNE_TOKEN
from utils.multiprocess import MultiProcess
from utils.load_data import LoadData

from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from qraft_data.data import Tag as QraftDataTag

from strategy_integration.components.integration.deep.deep_integration import (
    DeepIntegration,
)
from strategy_integration.components.models.deep_models import amom_model

from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores

#################################### 공통설정부분 ##########################################
NUM = 20


setting_params = {
    "identifier": f"amom_{NUM}_short_ver2",  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON"],
        "backtest_mode": True,
        "except_no_isin_code": False,
        "class_a_only": True,
        "pretend_monthend": False,
        "primary_issue": True,
    },
    "seed_config": {
        "training_model": True,
        "training_data_loader": None,
        "train_valid_data_split": None,
    },
    "csv_summary_save": True,
    "omniboard_summary_save": False,
    "tensorboard_summary_save": True,

    "cpu_count": 12
}

date_dict = {
    "date_from": "2015-12-31",
    "date_to": "2021-10-31",
    "rebalancing_terms": "M",
}

#################################### 배치메이커 ##########################################

# Preprocess input dataset
input_data = [
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
    ("trimmed_pce", ["high_level", "macro", "get_trimmed_mean_pce_rate"], [], {"change_to": "none"}),
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
    ("capa_util", ["fred", "capacity_utilization"]),
    ("unemployment", ["high_level", "macro", "get_unemployment_rate"]),
    ("snp500_vol", ["high_level", "macro", "get_snp500_vol"]),
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
    ("res_mom_12m_1m_0m", ["high_level", "equity", "get_res_mom"], [12, 1, 0]),
    ("res_mom_12m_2m_0m", ["high_level", "equity", "get_res_mom"], [12, 2, 0]),
    ("res_mom_12m_3m_0m", ["high_level", "equity", "get_res_mom"], [12, 3, 0]),
    ("res_mom_12m_4m_0m", ["high_level", "equity", "get_res_mom"], [12, 4, 0]),
    ("res_mom_12m_5m_0m", ["high_level", "equity", "get_res_mom"], [12, 5, 0]),
    ("res_mom_12m_6m_0m", ["high_level", "equity", "get_res_mom"], [12, 6, 0]),
    ("res_mom_12m_7m_0m", ["high_level", "equity", "get_res_mom"], [12, 7, 0]),
    ("res_mom_12m_8m_0m", ["high_level", "equity", "get_res_mom"], [12, 8, 0]),
    ("res_mom_12m_9m_0m", ["high_level", "equity", "get_res_mom"], [12, 9, 0]),
    ("res_mom_12m_10m_0m", ["high_level", "equity", "get_res_mom"], [12, 10, 0]),
    ("res_mom_12m_11m_0m", ["high_level", "equity", "get_res_mom"], [12, 11, 0]),
    ("res_mom_12m_12m_0m", ["high_level", "equity", "get_res_mom"], [12, 12, 0]),
    ("res_vol_1m", ["high_level", "equity", "get_realized_vol"], ["tr", 1]),
    ("res_vol_2m", ["high_level", "equity", "get_realized_vol"], ["tr", 2]),
    ("res_vol_3m", ["high_level", "equity", "get_realized_vol"], ["tr", 3]),
    ("res_vol_4m", ["high_level", "equity", "get_realized_vol"], ["tr", 4]),
    ("res_vol_5m", ["high_level", "equity", "get_realized_vol"], ["tr", 5]),
    ("res_vol_6m", ["high_level", "equity", "get_realized_vol"], ["tr", 6]),
    ("res_vol_7m", ["high_level", "equity", "get_realized_vol"], ["tr", 7]),
    ("res_vol_8m", ["high_level", "equity", "get_realized_vol"], ["tr", 8]),
    ("res_vol_9m", ["high_level", "equity", "get_realized_vol"], ["tr", 9]),
    ("res_vol_10m", ["high_level", "equity", "get_realized_vol"], ["tr", 10]),
    ("res_vol_11m", ["high_level", "equity", "get_realized_vol"], ["tr", 11]),
    ("res_vol_12m", ["high_level", "equity", "get_realized_vol"], ["tr", 12]),
    ("mv", ["high_level", "equity", "get_monthly_market_value"]),
    ("gpa_trailing", ["high_level", "equity", "get_gpa"], [], {'as_trailing':True}),
    ("gpm_trailing", ["high_level", "equity", "get_gp_to_market"], [], {'as_trailing': True}),
    ("ebitdaev_trailing", ["high_level", "equity", "get_ebitda_to_ev"], [], {'as_trailing': True}),
    ("btm", ["high_level", "equity", "get_book_to_market"]),
    ("sector", ["high_level", "equity", "get_historical_gics"], ["sector"]),
]

#################################### integration ##########################################

integrated_model_params = {
    "training_all_at_once": False,
    "reinitialization": 2,
    "models": [
        {
            "name": amom_model.AmomModel.__name__,
            "fill_nan_value": float("NaN"),
            "inputs": ["x"],
            "targets": ["y"],
            "forwards": [],
        }
    ],
    "epochs": 5,
    "forwards_at_epoch": [],
    "forward_names": [],
    "early_stopping_start_after": 200,
    "early_stopping_interval": 200,
    "adversarial_info": {
        "adversarial_use": False,
        "adversarial_alpha": 0.0,
        "adversarial_weight": 0.0,
    },
    "path": DATA_DIR,
    "save_all_epochs": False,
    "weight_share": False,
    "batch_update": ["optimizer"],
    "epoch_update": [],
    "batch_size": 64,
    "validation_ratio": 0.0,
    "shuffle": False,
}


neptune_dict = {
    "use_neptune": False,  # 테스트기간동안에는 잠시 False로 해두겠습니다.
    "user_id": "amom",
    "project_name": "amom",
    "exp_id": "amom-0228",
    "exp_name": "amom",
    "description": "amom",
    "tags": ["amom"],
    "hparams": {},
    "token": NEPTUNE_TOKEN,
}


def amom_running(lc, shapes_dict, d_inference):
    di = DeepIntegration(
        identifier=setting_params["identifier"],
        sub_identifier=d_inference.strftime("%Y-%m-%d"),
        dataset=lc,
        data_shape=shapes_dict,
        integrated_model_params=integrated_model_params,
        setting_params=setting_params,
    )

    split_item: bm.splitting.NonSplit = lc.training_split(bm.splitting.non_split)

    reinitialization = integrated_model_params.get('reinitialization')
    for i in range(reinitialization):
        early_stopping, best_loss = di.initialize_train()
        early_stopping_flag = False

        for epoch in range(1, integrated_model_params["epochs"] + 1):
            if early_stopping_flag:
                break
            train_step_info_list = []
            train_stack_forwards = {}
            training_flags = False

            di._set_all_train_mode()

            for meta, data in lc.split_of(split_item.ALL).training_iterate(
                batch_size=integrated_model_params["batch_size"],
                inclusive=True,
                same_time=False,
                probabilistic_sampling=False,
                drop_last_if_not_probabilistic_sampling=True,
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
                f'training: {epoch}/{integrated_model_params["epochs"]} ({best_loss[0]:.4f})',
                end="\r",
            )

            di._set_all_validation_mode()
            for meta, data in lc.inference_iterate():
                sec_keys = meta["SECURITY"]

                if integrated_model_params["save_all_epochs"]:
                    di.infer_with_all_epochs(epoch, data, sec_keys)

                if (epoch == integrated_model_params["epochs"] or early_stopping_flag) and i == (reinitialization-1):
                    di.infer_with_all_epochs(None, data, sec_keys)
                    di.summary_after_infer(
                        epoch, integrated_model_params["epochs"], best_loss
                    )
            di.run_after_epoch()

        if i < reinitialization-1:
            di.reset()


if __name__ == "__main__":
    # universe = Universe(**setting_params["kirin_config"])
    #
    # api = get_kirin_api(universe)
    # _back_length = 18
    # _input_length = 12
    #
    # load_dataset = LoadData(path= Path(DATA_DIR) / "kirin_dataset", universe=universe)
    #
    # with mp.Pool(10) as pool:
    #     res = pool.starmap(load_dataset.call_if_not_loaded, input_data)
    #
    # sector_values = load_dataset.call_if_not_loaded("sector_values", ["compustat", "read_sql"], ["SELECT giccd FROM r_giccd WHERE gictype='GSECTOR';"])
    # sector_values = sector_values.values.reshape(-1)
    # for i, qdata in enumerate(res):
    #     if qdata.get_tag() == QraftDataTag.INDEX.value:
    #         qdata = qdata.rolling(18).winsorize((0.01, 0.99), pct=True).zscore()
    #
    #     if qdata.get_tag() == QraftDataTag.EQUITY.value:
    #         if qdata.name == "sector":
    #             qdata = qdata.one_hot(sector_values)
    #         else:
    #             qdata = qdata.winsorize((0.01, 0.99), pct=True).zscore()
    #
    #     input_data[i] = qdata
    #
    # # preprocess output
    # output = load_dataset.call_if_not_loaded("pr_1m", ["high_level", "equity", "get_monthly_price_return"], [1, 0])
    # output = output.winsorize((0.01, 0.99), pct=True).zscore()
    #
    # # Masking training & inferenc filter
    # filter_training = load_dataset.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    # filter_training._data = filter_training._data.notna()
    #
    # filter_mv = load_dataset.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    # filter_mv = filter_mv.rank(ascending=False, pct=True) <= 0.2
    #
    # gpa = load_dataset.call_if_not_loaded(
    #         "gpa_trailing", ["high_level", "equity", "get_gpa"], [], {'as_trailing':True}).masked_by(filter_mv)
    # mom_12m_1m = load_dataset.call_if_not_loaded("mom_12m_1m", ["high_level", "equity", "get_mom"], ["pi", 12, 1]).masked_by(filter_mv)
    # gpa = gpa.zscore()
    # mom_12m_1m = mom_12m_1m.zscore()
    # v = gpa + mom_12m_1m
    # filter_infer = v.rank(ascending=False, pct=False) <= 100
    #
    # index, columns = bm.checking.check_equal_index_and_columns(input_data + [output, filter_training, filter_infer])
    #
    # input_x = bm.DataBinder(
    #     data_list=input_data,
    #     training_filter=filter_training,
    #     inference_filter=filter_infer,
    #     length=_input_length,
    #     is_input=True,
    #     max_nan_ratio=0.0,
    #     aug=None,
    # )
    # output_y = bm.DataBinder(
    #     data_list=[output],
    #     training_filter=filter_training,
    #     inference_filter=filter_infer,
    #     length=1,
    #     is_input=False,
    #     max_nan_ratio=0.0,
    #     aug=None,
    # )
    # sbm = bm.SecurityBatchMaker(
    #     save_path=(
    #         Path(DATA_DIR) / setting_params["identifier"] / "batch_dataset"
    #     ).as_posix(),
    #     index=index,
    #     columns=columns,
    #     data_map={
    #         "x": input_x,
    #         "y": output_y,
    #     },
    #     max_cache="15GB",
    #     probability=None,
    # )
    #
    # dates_inference = pd.date_range(
    #     start=date_dict["date_from"],
    #     end=date_dict["date_to"],
    #     freq=date_dict["rebalancing_terms"],
    # ).to_pydatetime()
    #
    # shapes_dict = sbm.get_sample_shapes()
    #
    # run_proc = MultiProcess()
    # run_proc.cpu_count(max_count=setting_params['cpu_count'])
    #
    # for d_inference in dates_inference:
    #     with sbm.local_context(
    #         inference_dates=d_inference, length=_back_length - _input_length
    #     ) as lc:
    #         proc = mp.Process(target=amom_running, args=(lc, shapes_dict, d_inference))
    #         proc.start()
    #         run_proc.run_process(proc)
    #         time.sleep(0.1)
    #
    # run_proc.final_process()
    #
    # st = Strategy(
    #     data_dir=DATA_DIR,
    #     identifier=setting_params['identifier'],
    #     kirin_config=setting_params['kirin_config'],
    #     date_from=date_dict['date_from'],
    #     date_to=date_dict['date_to'],
    #     rebalancing_terms=date_dict['rebalancing_terms'],
    #
    #     long_picking_config = picks.picking_by_signal("mu", False, 1, NUM, ascending=False),
    #     long_weighting_config=(
    #         weights.dynamic_weight("0.5*mw + 0.5*ew"),
    #         weights.optimal_weight(
    #             kirin_config=setting_params["kirin_config"],
    #             loss_type="MSE",
    #             max_weight=0.10,
    #             threshold_weight=0.05,
    #             bound_sum_threshold_weight=0.4,
    #             bound_gics={"sector": 0.5, "industry": 0.24},
    #             bound_financials_sector={"40": 0.048},
    #         ),
    #     ),
    #
    #     factor=True,
    #     market_percentile=0.2,
    #
    #     gics=True,
    #     gics_level=["sector"],
    # )
    #
    # st.backtest()

    cp = Comparison(
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        kirin_config=setting_params['kirin_config'],
        # performance_fps=list(st.get_performance_fps()),
        performance_fps=["/raid/sr-storage/amom_20_short_ver2"],
        performance_names=["amom20_short_ver1"],
        # performance_names=list(st.get_performance_names()),
        standard_benchmarks=["U:SPMO", "U:MTUM"],
        comparison_periods=[],
        final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
        neptune_params=neptune_dict
    )

    cp.compare()
