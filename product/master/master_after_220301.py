import sys
import time
from pathlib import Path
import copy

sys.path.append(".")

import pandas as pd
import torch.multiprocessing as mp

from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from qraft_data.data import Tag as QraftDataTag

import batch_maker as bm

from strategy_integration.components.integration.deep.deep_integration import DeepIntegration
from strategy_integration.components.models.deep_models.multi_source_attention_model import MultiSourceAttentionModel

from utils.load_data import LoadData
from utils.multiprocess import MultiProcess

from paths import DATA_DIR, NEPTUNE_TOKEN
from product.master.baskets import *
from product.master.five_factors import LoadFiveFactor
import product.master.infer_filters as infer_filters


training_universe = sys.argv[1].strip()
input_factors = sys.argv[2].strip()
year = int(sys.argv[3].strip())

assert training_universe in ("it", "nasdaq")
assert input_factors in ("five_factors", "qraft_factors")

EPOCHS = 23
BATCH_SIZE = 32
CROSS_NUM_ITER = 100
CROSS_NUM_UNIVERSE = 256
TRAINING_LENGTH = 12*year
SAMPLE_LENGTH = 36

ROOT = "master_test_220301"
IDENTIFIER = ROOT + f"{training_universe}_{input_factors}_{year}years"
CACHE_PATH = f"cache_{ROOT}"
DATASET_PATH = f"dataset_{ROOT}"
NASDAQ_CACHE_PATH = f"cache_{ROOT}_nasdaq_only"
NASDAQ_DATASET_PATH = f"dataset_{ROOT}_nasdaq_only"
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

DATE_FROM = "2009-12-31"
DATE_TO = "2022-01-31"

neptune_dict = {
    "use_neptune": False,  # 넵튠을 사용하려면 True 로 표시합니다.
    "user_id": "jayden",  # USER_ID는 당분간 저로 고정합니다
    "project_name": "jayden",  # 프로젝트는 jayden, tei, aiden 으로 만들어 뒀습니다
    "exp_name": "",  # 실험의 이름 필수는 아닙니다
    "description": "",  # 실험의 설명 필수는 아닙니다
    "hparams": {},  # 저장하고 싶은 하이퍼 파라미터를 딕셔너리로, 굳이 안넣어줘도 소스코드가 저장되어 당시의 셋팅을 확인가능합니다.
    "tags": [],  # 마스터 프로젝트에서는 태그를 변경하지 않습니다
    "exp_id": ["NEW"],  # 마스터 프로젝트에서는 EXP_ID를 변경하지 않습니다
    "token": NEPTUNE_TOKEN,  # 키는 고정입니다
}

setting_params = {
    "identifier": IDENTIFIER,  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / CACHE_PATH).as_posix(),
        "use_sub_server": True,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON", "ADR"],
        "backtest_mode": True,
        "except_no_isin_code": False,
        "class_a_only": False,
        "primary_issue": False,
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

integrated_model_params = {
    "training_all_at_once": False,
    "reinitialization": 1,
    "models": [
        {
            "name": MultiSourceAttentionModel.__name__,
            "inputs": ["x"],
            "targets": ["y"],
            "forwards": [],
            "cross_units": {
                "x": [64, 32],
            },
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
    # DeepIntegration instance 생성
    di = DeepIntegration(
        identifier=setting_params["identifier"],
        sub_identifier=inference_date.strftime("%Y-%m-%d"),
        dataset=None,
        data_shape=dict_of_shapes,
        integrated_model_params=integrated_model_params,
        setting_params=setting_params,
    )
    # context list를 만듬
    training_lc = context_list[0]
    inference_lc_list = context_list

    # split_item 은 1:training 이거나 2: validation 인 tag임
    # training_lc 는 local_context객체로, training_split을 통해 나눠줌
    split_item: bm.splitting.TrainingValidationSplit = training_lc.training_split(
        bm.splitting.training_validation_split,
        validation_ratio=integrated_model_params["validation_ratio"],
        shuffle=integrated_model_params["shuffle"],
    )

    # early stopping 객체 초기화
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
        print(f'training: {epoch}/{integrated_model_params["epochs"]} ({best_loss[0]:.4f})')

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
        print(f'validation : {epoch}/{integrated_model_params["epochs"]} ({best_loss[0]:.4f})')

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
    universe = Universe(**setting_params["kirin_config"])
    api = get_kirin_api(universe)

    loader = LoadData(path=Path(DATA_DIR) / DATASET_PATH, universe=universe)

    y = loader.call_if_not_loaded("tr_1m_0m", ["high_level", "equity", "get_monthly_price_return"], [1, 0])
    y = 2 * y.rank(pct=True) - 1

    if input_factors == "five_factors":    
        five_loader = LoadFiveFactor(path=Path(DATA_DIR) / DATASET_PATH, universe=universe)
        momentum = five_loader.call_if_not_loaded("five_factors_momentum")
        value = five_loader.call_if_not_loaded("five_factors_value")
        size = five_loader.call_if_not_loaded("five_factors_size")
        quality = five_loader.call_if_not_loaded("five_factors_quality")
        low_vol = five_loader.call_if_not_loaded("five_factors_low_vol")

        inputs = [momentum, value, size, quality, low_vol]
    
    else:
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
            ("ia_mv", ["high_level", "equity", "get_linear_assumed_intangible_asset_to_market_value"],),
            ("ae_m", ["high_level", "equity", "get_advertising_expense_to_market"]),
            ("ia_ta", ["high_level", "equity", "get_linear_assumed_intangible_asset_to_total_asset"],),
            ("rc_a", ["high_level", "equity", "get_rnd_capital_to_asset"]),
            ("r_s", ["high_level", "equity", "get_rnd_to_sale"]),
            ("r_a", ["high_level", "equity", "get_rnd_to_asset"]),
        ]
        with mp.Pool(8) as pool:
            raw_inputs = pool.starmap(loader.call_if_not_loaded, input_data)

        inputs = []
        for raw_data in raw_inputs:
            if raw_data.get_tag() == QraftDataTag.INDEX.value:
                raw_data = raw_data.rolling(SAMPLE_LENGTH).zscore()
            if raw_data.get_tag() == QraftDataTag.EQUITY.value:
                raw_data = 2 * raw_data.rank(pct=True) - 1
            inputs.append(raw_data)

    # training filter
    base_filter = infer_filters.get_base_filter(loader)

    if training_universe == "it":
        training_filter = infer_filters.get_it_filter(base_filter, loader)
    else:
        training_filter = infer_filters.get_nasdaq_filter(
            base_filter, 
            setting_params["kirin_config"], 
            (Path(DATA_DIR) / NASDAQ_CACHE_PATH).as_posix(), 
            (Path(DATA_DIR) / NASDAQ_DATASET_PATH).as_posix()
        )

    basket2_filter = infer_filters.get_basket2_filter(training_filter)
    it_all_filter = infer_filters.get_it_filter(training_filter, loader)
    it_top60_filter = infer_filters.get_it_top60_filter(training_filter, loader)

    probability = training_filter.copy()
    probability._data = probability._data.astype(float)

    # inference filter
    infer_filter_list = [basket2_filter, it_all_filter, it_top60_filter]
    name_list = ["BASKET2", "IT_ALL", "IT_TOP60"]
    
    # Check input, output and filter dataset has same index and columns
    index, columns = bm.checking.check_equal_index_and_columns(inputs + infer_filter_list + [training_filter])

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
            save_path=(Path(DATA_DIR) / setting_params["identifier"] / "batch_dataset" / name).as_posix(),
            index=index,
            columns=columns,
            data_map={"x": binder_x, "y": binder_y},
            max_cache="30GB",
            probability=probability,
        )
        sbm_list.append(sbm)

    date_list = pd.date_range(
        start=date_dict["date_from"],
        end=date_dict["date_to"],
        freq=date_dict["rebalancing_terms"],
    ).to_pydatetime()
    
    target_date_list = []
    for d in date_list:
        file_path = Path(DATA_DIR) / IDENTIFIER / "infer" / f"infer_at_{d.strftime('%Y-%m-%d')}.csv"
        if not file_path.is_file():
            target_date_list.append(d)
    
    if not target_date_list:
        exit(0)

    shapes_dict = sbm_list[0].get_sample_shapes()

    run_proc = MultiProcess()
    run_proc.cpu_count(max_count=setting_params['cpu_count'])
    for d_inference in target_date_list:
        lc_list = []
        length_list = [TRAINING_LENGTH] + [0] * (len(sbm_list) - 1)
        for sbm, length in zip(sbm_list, length_list):
            lc = sbm.local_context(inference_dates=d_inference, length=length)
            lc_list.append(lc)
        
        proc = mp.spawn(run_model, args=(name_list, lc_list, shapes_dict, d_inference), join=False).processes[0]
        run_proc.run_process(proc)

    run_proc.final_process()
    exit(1)
