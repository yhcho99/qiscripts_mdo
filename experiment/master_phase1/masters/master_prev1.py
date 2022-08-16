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
from qraft_data.data import QraftData

from strategy_integration.components.integration.deep.deep_integration import (
    DeepIntegration,
)
from strategy_integration.components.models.deep_models import attention_model

from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores


#################################### 공통설정부분 ##########################################
setting_params = {
    "identifier": "softbank_test",  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "softbank_kirin_cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON", "ADR"],
        "backtest_mode": False,
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

    "cpu_count": 24
}

date_dict = {
    "date_from": "2020-12-31",
    "date_to": "2021-02-28",  #
    "rebalancing_terms": "M",
}


#################################### 배치메이커 ##########################################


# Preprocess input dataset
input_data = [
    ("pr_1m_0m", ["high_level", "equity", "get_monthly_price_return"], [1, 0], dict()),
    ("mv", ["high_level", "equity", "get_monthly_market_value"], [], dict()),
    ("btm", ["high_level", "equity", "get_book_to_market"], [], dict()),
    (
        "mom_12m_1m",
        ["high_level", "equity", "get_monthly_price_return"],
        [12, 1],
        dict(),
    ),
    ("ram_12m_0m", ["high_level", "equity", "get_ram"], ["pi", 12], {}),
    ("vol_3m", ["high_level", "equity", "get_monthly_volatility"], [3], {}),
    ("res_mom_12m_1m_0m", ["high_level", "equity", "get_res_mom"], [12, 1, 0], {}),
    ("res_vol_6m_3m_0m", ["high_level", "equity", "get_res_vol"], [6, 3, 0], {}),
    ("at", ["high_level", "equity", "get_asset_turnover"], [], {}),
    ("gpa", ["compustat", "custom_api", "get_gpa"], [], {}),
    ("rev_surp", ["high_level", "equity", "get_revenue_surprise"], [], {}),
    ("cash_at", ["high_level", "equity", "get_cash_to_asset"], [], {}),
    ("op_lev", ["high_level", "equity", "get_operating_leverage"], [], {}),
    ("roe", ["high_level", "equity", "get_roe"], [], {}),
    (
        "std_u_e",
        ["high_level", "equity", "get_standardized_unexpected_earnings"],
        [],
        {},
    ),
    ("ret_noa", ["high_level", "equity", "get_return_on_net_operating_asset"], [], {}),
    ("etm", ["high_level", "equity", "get_earnings_to_market"], [], {}),
    (
        "ia_mv",
        ["high_level", "equity", "get_linear_assumed_intangible_asset_to_market_value"],
        [],
        {},
    ),
    ("ae_m", ["high_level", "equity", "get_advertising_expense_to_market"], [], {}),
    (
        "ia_ta",
        ["high_level", "equity", "get_linear_assumed_intangible_asset_to_total_asset"],
        [],
        {},
    ),
    ("rc_a", ["high_level", "equity", "get_rnd_capital_to_asset"], [], {}),
    ("r_s", ["high_level", "equity", "get_rnd_to_sale"], [], {}),
    ("r_a", ["high_level", "equity", "get_rnd_to_asset"], [], {}),
    # ("gics", ['high_level', 'equity', 'get_historical_gics'], ['sector'], {}),
    ("t_2y", ["high_level", "macro", "get_us_treasury_2y"], [], {}),
    ("baa", ["high_level", "macro", "get_moodys_baa_corporate_bond_yield"], [], {}),
    ("t_10y", ["high_level", "macro", "get_us_treasury_10y"], [], {}),
    ("snp500_vol", ["high_level", "macro", "get_snp500_vol"], [], {}),
    ("snp500_pe", ["high_level", "macro", "get_snp_pe"], []),
    ("snp500_pr", ["high_level", "index", "get_snp500_momentum"], [], {"periods": 1}),
    (
        "tmpr",
        ["high_level", "macro", "get_trimmed_mean_pce_rate"],
        [],
        {"change_to": "none"},
    ),
    ("unemploy", ["high_level", "macro", "get_unemployment_rate"], [], {}),
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
    "epochs": 1,
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
    "batch_size": 256,
    "validation_ratio": 0.2,
    "shuffle": True,
}

#################################### simulation ##########################################

strategy_dict = {
    "short": False,  # short을 할지의 여부입니다
    "long_amount": 1.0,  # long포트폴리오의 총 비중입니다. ex) longonly의 경우 1.0  , 130/30의 경우 1.3
    "short_amount": 0.0,  # short포트폴리오의 총 비중입니다. ex) longonly의 경우 0. , 130/30의 경우 -0.3
    "long_picking_config": picks.picking_by_signal(
        "out", False, 1, 13, ascending=False
    ),  # 롱포트폴리오 뽑는방법
    "short_picking_config": picks.picking_by_signal(
        "out", False, 1, 30, ascending=True
    ),  # 숏포트폴리오 뽑는방법
    "long_weighting_config": (
        weights.test_weight(0.995, 13, 1),
    ),
    "short_weighting_config": weights.market_weight(),  # 숏 종목 비중 주는 방법
    "weight_adjusting_unitdate": False,  # 리밸런싱 시점에 관계없이 매 시점마다 weight 구하는 방법입니다
    "backtest_daily_out": False,  # 월별로 구성된 포트폴리오를 일별로 확장하여 백테스트를 할 것인지 여부
    "backtest_daily_out_lag": [0, 1],  #
    ########### 포트폴리오에서로 구할때는 위에 것들은 따로 설정해줄 필요가 없습니다 ############
    "from_which": "infer",  # infer: infer 데이터로부터 strategy가 진행됩니다. portfolio : universe.csv와 weight.csv가 존재할 경우입니다
    "factor": False,  # factor 와 관련된 백테스트를 할지 여부
    "save_factor": False,  # factor와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    "market_percentile": 0.2,  # 시가총액 상위 몇 %의 주식을 볼지
    "gics": True,  # gics 와 관련된 백테스트를 할지 여부
    "save_gics": True,  # gics와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    "gics_level": ["sector"],  # gics 레벨 결정
}

# performance_fps 와 performance_names의 경우 strategy를 먼저 실행했을 경우
# get_performance_fps() , get_performance_names() 함수를 통해 얻을 수 있고
# comparison부터 시작할 경우엔 fps의 경우 폴더명이나 file_path 를 적어주면 됩니다

comparison_dict = {
    "performance_fps": [],  # identifier와 동일한 값 혹은, 전체 performance file paths
    "performance_names": [],  # 각 퍼포먼스 별 별칭
    "standard_benchmarks": ["U:SPY", "@QQQ"],  # 벤치마크로 삼을 U:SPY
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
    _back_length = 48
    _input_length = 36

    load_dataset = LoadData(path=DATA_DIR + '/softbank_dataset', universe=universe)

    with mp.Pool(10) as pool:
        res = pool.starmap(load_dataset.call_if_not_loaded, input_data)

    sector_values = load_dataset.call_if_not_loaded(
        "sector_values",
        ["compustat", "read_sql"],
        {"SELECT giccd FROM r_giccd WHERE gictype='GSECTOR';"},
    )
    sector_values = sector_values.values.reshape(-1)
    for i, qdata in enumerate(res):
        if qdata.get_tag() == QraftDataTag.INDEX.value:
            qdata = qdata.rolling(_input_length).zscore()

        if qdata.get_tag() == QraftDataTag.EQUITY.value:
            if qdata.name == "gics":
                qdata = qdata.one_hot(sector_values)
            else:
                qdata = qdata.rank(ascending=True, pct=True)

        input_data[i] = qdata

    # preprocess mv & output
    additional_mv = load_dataset.call_if_not_loaded(
        "mv", ["compustat", "get_monthly_market_value"]
    )
    output = load_dataset.call_if_not_loaded(
        "pr_1m_0m", ["high_level", "equity", "get_monthly_price_return"], 1, 0
    )

    additional_mv = additional_mv.minmax()
    output = output.winsorize((0.01, 0.99), pct=True).zscore()

    # Masking training & inference filter
    market_value = load_dataset.call_if_not_loaded(
        "mv", ["high_level", "equity", "get_monthly_price_data"]
    )
#     filter_training = market_value.rank(pct=True, ascending=False) <= 0.2
    filter_training = market_value._data.notna()
    filter_training = QraftData("filter_training", filter_training)
    filter_inference = market_value._data.copy()
    filter_inference.loc[:] = False
    gvkey_iids = [
        '020530_90', '017874_01', # '037527_01', 
        '012540_01', '160329_03', '064768_01',
        '170617_01', '012141_01', '147579_01', 
        '185419_01', '024616_01', '157855_01',
        '201395_90', # '035729_01', 
        '117768_01',
    ]
    filter_inference.loc[:, gvkey_iids] = True
    filter_inference = QraftData("filter_inference", filter_inference)

    # Check input, output and filter dataset has same index and columns
    index, columns = bm.checking.check_equal_index_and_columns(
        input_data + [additional_mv, output, filter_training]
    )

    input_x = bm.DataBinder(
        data_list=input_data,
        training_filter=filter_training,
        inference_filter=filter_inference,
        length=_input_length,
        is_input=True,
        max_nan_ratio=0.99,
        aug=None,
    )
    input_mv = bm.DataBinder(
        data_list=[additional_mv],
        training_filter=filter_training,
        inference_filter=filter_inference,
        length=1,
        is_input=True,
        max_nan_ratio=0.99,
        aug=None,
    )
    output_y = bm.DataBinder(
        data_list=[output],
        training_filter=filter_training,
        inference_filter=filter_inference,
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
        max_cache="15GB",
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

    comparison_dict["performance_fps"] = st.get_performance_fps()
    comparison_dict["performance_names"] = st.get_performance_names()

    cp = Comparison(
        data_dir=DATA_DIR,
        setting_params=setting_params,
        neptune_params=neptune_dict,
        **comparison_dict,
    )
    cp.compare()
