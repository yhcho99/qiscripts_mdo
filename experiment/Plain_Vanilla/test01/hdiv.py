import sys
import time
sys.path.append("")
sys.path.append("/home/sronly/Projects/qiscripts")
sys.path.append("/home/hyungyunjeon/QRAFT/git_clones/qiscripts")
import pandas as pd
import numpy as np
import torch
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
from strategy_integration.components.models.deep_models import hdiv_model

from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores


#################################### 공통설정부분 ##########################################
setting_params = {
    "identifier": "Plain_Vanilla_mixedHDIV_test001",  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
        "use_sub_server": False,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON"],
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

    "cpu_count": 4
}

date_dict = {
    "date_from": "2009-12-31",
    "date_to": "2021-10-31",
    "rebalancing_terms": "M",
}

#################################### 배치메이커 ########################################
# Preprocess input dataset
input_data = [
    ("outperform_mom_1m_0m", ["high_level", "equity", "get_outperform_mom"], ["pi", 1, 0]),
    ("mv", ["compustat", "get_monthly_market_value"]),
    ("mom_3m_0m", ["high_level", "equity", "get_mom"], ["pi", 3, 0]),
    ("mom_6m_0m", ["high_level", "equity", "get_mom"], ["pi", 6, 0]),
    ("gpa", ["compustat", "custom_api", "get_gpa"]),
    ("gpm", ["compustat", "custom_api", "get_gp_to_market"]),
    ("snp500_pr", ["high_level", "index", "get_snp500_momentum"], [], {"periods": 1}),
    ("fred_ff", ["fred", "ff"]),
]


#################################### integration ##########################################

integrated_model_params = {
    "training_all_at_once": False,
    "reinitialization": 1,
    "models": [
        {
            "name": hdiv_model.HdivModel.__name__,
            "inputs": ["x"],
            "targets": ["y"],
            "forwards": [],
        },
    ],
    "epochs": 200,
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
    "batch_size": 128,
    "validation_ratio": 0.0,
    "shuffle": False,
}

neptune_dict = {
    "use_neptune": False,  # 테스트기간동안에는 잠시 False로 해두겠습니다.
    "user_id": "jayden",
    "project_name": "hdiv",
    "exp_id": "hdiv-14",
    "exp_name": "hi",
    "description": "This experiments are the most important",
    "tags": ["nvq", "attention"],
    "hparams": {},
    "token": NEPTUNE_TOKEN,
}


def hdiv_running(lc, shapes_dict, d_inference):
    di = DeepIntegration(
        identifier=setting_params["identifier"],
        sub_identifier=d_inference.strftime("%Y-%m-%d"),
        dataset=lc,
        data_shape=shapes_dict,
        integrated_model_params=integrated_model_params,
        setting_params=setting_params,
    )
    split_item: bm.splitting.NonSplit = lc.training_split(bm.splitting.non_split)
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
            replace_output = []
            for label in data["y"]:
                label[label > 10] = 10
                label[label < -10] = -10
                new_label = label[~label.isnan()].mean()
                if new_label.isnan():
                    new_label = torch.tensor(0).float()
                replace_output.append(new_label.numpy())
            replace_output = np.array(replace_output)

            data["y"] = torch.from_numpy(np.array(replace_output).reshape(-1, 1, 1))
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

            elif epoch == integrated_model_params["epochs"] or early_stopping_flag:
                di.infer_with_all_epochs(None, data, sec_keys)
                di.summary_after_infer(
                    epoch, integrated_model_params["epochs"], best_loss
                )
        di.run_after_epoch()


if __name__ == "__main__":
    universe = Universe(**setting_params["kirin_config"])

    api = get_kirin_api(universe)

    _back_length = 12
    _input_length = 6

    load_dataset = LoadData(path= Path(DATA_DIR) / "kirin_dataset", universe=universe)

    with mp.Pool(8) as pool:
        res = pool.starmap(load_dataset.call_if_not_loaded, input_data)

    for i, qdata in enumerate(res):
        if qdata.get_tag() == QraftDataTag.INDEX.value:
            qdata = qdata.rolling(12).zscore()

        if qdata.get_tag() == QraftDataTag.EQUITY.value:
            qdata = qdata.zscore().winsorize(clip_val=10.0, pct=False)

        input_data[i] = qdata

    # Preprocess output dataset
    output = load_dataset.call_if_not_loaded(
        "outperform_mom_1m_0m", ["high_level", "equity", "get_outperform_mom"], ["pi", 1, 0])
    output = output.winsorize(clip_val=0.05, pct=False).minmax()
    index, columns = bm.checking.check_equal_index_and_columns(input_data + [output])

    # Masking training_filter
    data = load_dataset.call_if_not_loaded("mv", ["compustat", "get_monthly_market_value"])
    filter_mv = data.rank(ascending=False, pct=True) <= 0.19

    # Masking inference_filter
    div = load_dataset.call_if_not_loaded("div", ["high_level", "equity", "get_dividend_yield"]).masked_by(filter_mv).zscore()
    gpa = load_dataset.call_if_not_loaded("gpa", ["high_level", "equity", "get_gpa"]).masked_by(filter_mv).zscore()
    gpm = load_dataset.call_if_not_loaded("gpm", ["high_level", "equity", "get_gp_to_market"]).masked_by(filter_mv).zscore()
    mom_1m_1m = load_dataset.call_if_not_loaded("mom_1m_1m", ["high_level", "equity", "get_mom"], ["pi", 1, 0]).masked_by(filter_mv).zscore()
    hdiv_factor = load_dataset.call_if_not_loaded("hdiv_factor", ["high_level", "equity", "get_hdiv_factor"]).masked_by(filter_mv).zscore()

    new_value = div + gpa + gpm + mom_1m_1m + hdiv_factor
    filter_infer = new_value.zscore().rank(ascending=False, pct=False) <= 150

    input_x = bm.DataBinder(
        data_list=input_data,
        training_filter=filter_mv,
        inference_filter=filter_infer,
        length=_input_length,
        is_input=True,
        max_nan_ratio=0.0,
        aug=None,
    )
    output_y = bm.DataBinder(
        data_list=[output],
        training_filter=filter_mv,
        inference_filter=filter_infer,
        length=3,
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
            proc = mp.Process(target=hdiv_running, args=(lc, shapes_dict, d_inference))
            proc.start()
            run_proc.run_process(proc)
            time.sleep(0.1)

    run_proc.final_process()

    st = Strategy(
        kirin_config=setting_params['kirin_config'],
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        date_from=date_dict['date_from'],
        date_to=date_dict['date_to'],
        rebalancing_terms=date_dict['rebalancing_terms'],
        long_picking_config=picks.picking_by_signal("mu/variance**0.5", False, 1, 50, ascending=False),
        long_weighting_config=(
            weights.market_weight(),
            weights.optimal_weight(
                kirin_config=setting_params["kirin_config"],
                loss_type="MSE",
                max_weight=0.08,
                threshold_weight=0.05,
                bound_sum_threshold_weight=0.4,
                bound_gics={"sector": 0.5, "industry": 0.24},
                bound_financials_sector={"40": 0.048},
            ),
        ),
        backtest_daily_out=False,
        factor=True,
        gics=True,
    )
    st.backtest()

    cp = Comparison(
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        kirin_config=setting_params['kirin_config'],
        performance_fps=list(st.get_performance_fps()),
        performance_names=list(st.get_performance_names()),
        standard_benchmarks=["U:SPYD", "U:VYM"],
        comparison_periods=[],
        final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
        neptune_params=neptune_dict
    )

    cp.compare()