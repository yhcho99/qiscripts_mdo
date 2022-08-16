
import sys
import time
sys.path.append("..")
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
from utils.master_universe import get_master_universe_gvkey

from strategy_integration.components.integration.deep.deep_integration import (
    DeepIntegration,
)
from strategy_integration.components.models.deep_models import attention_model

from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores


#################################### 공통설정부분 ##########################################


EPHOCS = 3

m_universe_list = ['universe1', 'universe2', 'universe3', 'simulation_universe1', 'simulation_universe2']
m_universe = m_universe_list[3]

_back_length = 48
_input_length = 36


setting_params = {
    "identifier": "master_day3_qrft_test01",  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
        "use_sub_server": False,
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

    "cpu_count": 8
}

date_dict = {
    "date_from": "1995-01-31",
    "date_to": "2021-03-31",
    "rebalancing_terms": "M",
}


#################################### 배치메이커 ##########################################
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

#################################### integration ##########################################

integrated_model_params = {
    "training_all_at_once": False,
    "reinitialization": 1,
    "models": [
        {
            "name": attention_model.AttentionModel.__name__,
            "cs_dims": 20,
            "ts_dims": 60,
            "bs_dims": 20,
            "dim_reduction": 2,
            "dr_ratio": 0.4,
            "amp": 1.0,
            "add_pick_loss": 8,
            "init_lr": 1e-4,
            "lr_decay": 0.91,
            "optimizer": "adam",
            "weight_decay": 3e-5,
            "inputs": ["x1", "x2"],
            "targets": ["y"],
            "forwards": [],
        },
    ],
    "epochs": EPHOCS,
    "forwards_at_epoch": [],
    "forward_names": [],
    "early_stopping_start_after": 5,
    "early_stopping_interval": 3,
    "adversarial_info": {
        "adversarial_use": False,
        "adversarial_alpha": 0.0,
        "adversarial_weight": 0.0,
    },
    "path": DATA_DIR,
    "save_all_epochs": False,
    "weight_share": False,
    "batch_update": ["optimizer"],
    "epoch_update": ["lr_scheduler"],
    "batch_size":1024,
    "validation_ratio": 0.2,
    "shuffle": True,
}

#################################### simulation ##########################################

strategy_dict = {
    "from_which": "infer",  # infer: infer 데이터로부터 strategy가 진행됩니다. portfolio : universe.csv와 weight.csv가 존재할 경우입니다
    
    "short": False,  # short을 할지의 여부입니다
    "short_amount": 0.0,  # short포트폴리오의 총 비중입니다. ex) longonly의 경우 0. , 130/30의 경우 -0.3
    "short_picking_config": picks.picking_by_signal("out", False, 1, None, ascending=True),  # 숏포트폴리오 뽑는방법
    "short_weighting_config": weights.market_weight(),  # 숏 종목 비중 주는 방법

    "long_amount": 1.0,  # long포트폴리오의 총 비중입니다. ex) longonly의 경우 1.0  , 130/30의 경우 1.3,
    "long_picking_config": picks.picking_by_signal("out", False, 1, None, ascending=False),  # 롱포트폴리오 뽑는방법 제한조건이 없을경우 None
    "long_weighting_config": (
        weights.dynamic_weight("0.5*mw + 0.5*ew"),
    ),

    ########### 이번 실험에서는 구할때는 아래를 따로 설정해줄 필요가 없습니다 ############
    "weight_adjusting_unitdate": False,  # 리밸런싱 시점에 관계없이 매 시점마다 weight 구하는 방법입니다
    "backtest_daily_out": False,  # 월별로 구성된 포트폴리오를 일별로 확장하여 백테스트를 할 것인지 여부
    "backtest_daily_out_lag": [0, 1],  #
    "factor": False,  # factor 와 관련된 백테스트를 할지 여부
    "save_factor": False,  # factor와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    "market_percentile": 0.2,  # 시가총액 상위 몇 %의 주식을 볼지
    "gics": False,  # gics 와 관련된 백테스트를 할지 여부
    "save_gics": False,  # gics와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    "gics_level": ["sector"],  # gics 레벨 결정
}



########################
########################
# 병렬 실험에 관련한 부분
########################
########################

def custom_tradevolume_weight():
    def _custom_tradevolume_weight(dt, signal, kirin_config, mv):


        w = signal.sort_values(ascending=True)

        return w

    return _custom_tradevolume_weight




strategy_mode = 3

if strategy_mode == 1:
    strategy_params = {
        'long_weighting_config': (
            weights.dynamic_weight('mw*(signal+0.000001)'),
                            ),
    }
    strategy_dict.update(strategy_params)
elif strategy_mode == 2:
    strategy_params = {
        'long_weighting_config': (
            weights.dynamic_weight('mw*((signal - 4.5/2 + 1.0)*(signal>=4.5/2) + (0.1)*(signal<4.5/2))'),
                             ),
    }
    strategy_dict.update(strategy_params)

elif strategy_mode == 3:
    strategy_params = {
        'long_weighting_config': (
            weights.dynamic_weight('mw*(1.0000001+signal)'),
                             ),
    }
    strategy_dict.update(strategy_params)

elif strategy_mode == 4:
    strategy_params = {
        'long_weighting_config': (
            weights.dynamic_weight('signal*mw'),                            ),
    }
    strategy_dict.update(strategy_params)




comparison_dict = {
    "performance_fps": [],  # identifier와 동일한 값 혹은, 전체 performance file paths
    "performance_names": [],  # 각 퍼포먼스 별 별칭
    "standard_benchmarks": ["S&PCOMP", "NASA100"],  # 벤치마크로 삼을 U:SPY
    "comparison_periods": [],  # 비교하고 싶은 기간
    "final_score": final_scores.annualized_return(
        exponential_decay_rate=None, total=False
    ),
}

neptune_dict = {
    "use_neptune": False,  # 테스트기간동안에는 잠시 False로 해두겠습니다.
    "user_id": "qrft",
    "project_name": "qrft",
    "exp_id": "qrft-0228",
    "exp_name": "qrft",
    "description": "qrft",
    "tags": ["qrft"],
    "hparams": {**comparison_dict},
    "token": NEPTUNE_TOKEN,
}


def qrft_running(lc, shapes_dict, d_inference):
    di = DeepIntegration(
        identifier=setting_params["identifier"],
        sub_identifier=d_inference.strftime("%Y-%m-%d"),
        dataset=lc,
        data_shape=shapes_dict,
        integrated_model_params=integrated_model_params,
        setting_params=setting_params,
    )

    split_item: bm.splitting.TrainingValidationSplit = lc.training_split(
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
        for meta, data in lc.split_of(split_item.TRAINING).training_iterate(
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

        training_flags = False
        di._set_all_validation_mode()
        for meta, data in lc.split_of(split_item.VALIDATION).training_iterate(
            batch_size=integrated_model_params["batch_size"],
            inclusive=True,
            same_time=False,
            probabilistic_sampling=False,
            drop_last_if_not_probabilistic_sampling=True,
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
            f'validation : {epoch}/{integrated_model_params["epochs"]} ({best_loss[0]:.4f})',
            end="\r",
        )

        if early_stopping.is_stopping(epoch):
            print(f"early stopping at {epoch} with loss {best_loss[0]:.4f}")
            early_stopping_flag = True

        di._set_all_validation_mode()
        for meta, data in lc.inference_iterate():
            sec_keys = meta["SECURITY"]

            if integrated_model_params["save_all_epochs"]:
                di.infer_with_all_epochs(epoch, data, sec_keys)

            if epoch == integrated_model_params["epochs"] or early_stopping_flag:
                di.infer_with_all_epochs(None, data, sec_keys)
                di.summary_after_infer(
                    epoch, integrated_model_params["epochs"], best_loss
                )


if __name__ == "__main__":

    universe = Universe(**setting_params["kirin_config"])
    api = get_kirin_api(universe)


    load_dataset = LoadData(path=DATA_DIR, universe=universe)

    with mp.Pool(8) as pool:
        res = pool.starmap(load_dataset.call_if_not_loaded, input_data)

    sector_values = load_dataset.call_if_not_loaded("sector_values", ["compustat", "read_sql"], ["SELECT giccd FROM r_giccd WHERE gictype='GSECTOR';"])
    sector_values = sector_values.values.reshape(-1)
    
    for i, qdata in enumerate(res):
        if qdata.get_tag() == QraftDataTag.INDEX.value:
            qdata = qdata.rolling(_input_length).zscore()
        
        if qdata.get_tag() == QraftDataTag.EQUITY.value:
            if qdata.name == "sector":
                qdata = qdata.one_hot(sector_values)
            else:
                qdata = qdata.rank(ascending=True, pct=True)
        
        input_data[i] = qdata

    # preprocess mv & output
    additional_mv = load_dataset.call_if_not_loaded("mv", ["compustat", "get_monthly_market_value"])
    output = load_dataset.call_if_not_loaded("pr_1m_0m", ["high_level", "equity", "get_monthly_price_return"], [1, 0])

    additional_mv = additional_mv.minmax()
    output = output.winsorize((0.01, 0.99), pct=True).zscore()

    # training filter
    training_filter_mv = load_dataset.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    training_filter_mv._data = training_filter_mv._data.notna()

    # inference filter
    infer_filter = load_dataset.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    master_gvkey_list = get_master_universe_gvkey(m_universe)
    infer_filter._data = infer_filter._data.isin(infer_filter._data[master_gvkey_list])

    # Check input, output and filter dataset has same index and columns
    index, columns = bm.checking.check_equal_index_and_columns(
        input_data + [additional_mv, output, infer_filter]
    )

    input_x = bm.DataBinder(
        data_list=input_data,
        training_filter=training_filter_mv,
        inference_filter=infer_filter,
        length=_input_length,
        is_input=True,
        max_nan_ratio=0.99,
        aug=None,
    )
    input_mv = bm.DataBinder(
        data_list=[additional_mv],
        training_filter=training_filter_mv,
        inference_filter=infer_filter,
        length=1,
        is_input=True,
        max_nan_ratio=0.99,
        aug=None,
    )
    output_y = bm.DataBinder(
        data_list=[output],
        training_filter=training_filter_mv,
        inference_filter=infer_filter,
        length=1,
        is_input=False,
        max_nan_ratio=0.0,
        aug=None,
    )
    sbm = bm.SecurityBatchMaker(
       save_path=(
           Path(DATA_DIR) / setting_params["identifier"] / "batch_dataset"
       ).as_posix(),
       index=index,
       columns=columns,
       data_map={
           "x": input_x,
           "mv": input_mv,
           "y": output_y,
       },
       max_cache="30GB",
       probability=None,
    )

    dates_inference = pd.date_range(
       start=date_dict["date_from"],
       end=date_dict["date_to"],
       freq=date_dict["rebalancing_terms"],
    ).to_pydatetime()

    shapes_dict = sbm.get_sample_shapes()

    run_proc = MultiProcess()
    run_proc.cpu_count(max_count=setting_params['cpu_count'])

    for d_inference in dates_inference:
        with sbm.local_context(
            inference_dates=d_inference, length=_back_length - _input_length
        ) as lc:
            proc = mp.Process(target=qrft_running, args=(lc, shapes_dict, d_inference))
            proc.start()
            run_proc.run_process(proc)
            time.sleep(0.1)

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

    cp = Comparison(
        data_dir=DATA_DIR,
        setting_params=setting_params,
        neptune_params=neptune_dict,
        **comparison_dict,
    )
    cp.compare()
