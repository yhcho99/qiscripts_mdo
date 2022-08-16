import sys
import time
sys.path.append("")
import pandas as pd
import numpy as np
import batch_maker as bm
import torch.multiprocessing as mp

from pathlib import Path
from paths import DATA_DIR, NEPTUNE_TOKEN
from utils.multiprocess import MultiProcess
from utils.load_data import LoadData
from utils.master_universe_gvkey import *

from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from qraft_data.data import Tag as QraftDataTag

from strategy_integration.components.integration.deep.deep_integration import DeepIntegration
from strategy_integration.components.models.deep_models import attention_model

from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores
from strategy_simulation.strategy.weighting import _add_mw_tw_ew




################################################################################################### 
############################################# 설정부분 ##############################################
################################################################################################### 

IDENTIFIER = "master_jh_qrft_exp8"
NEPTUNE_IDENTIFIER = f"{IDENTIFIER}"

DATE_FROM = "2021-03-31"
DATE_TO = "2021-04-30"

EPHOCS = 10
TRAINING_LENGTH = 36
SAMPLE_LENGTH = 12


def custom_weight(dt, signal, kirin_config, mv, tv):
    if len(signal) == 1:
        z = signal
    else:
        z = (signal - signal.mean()) / (signal.std() + 1e-6)
        
    signal = _add_mw_tw_ew(dt, signal, kirin_config, mv, tv)
    sig = 1 / (1 + np.exp(-z))
    sig = sig / sig.sum()
    return sig.sort_values(ascending=False)


strategy_dict = {
    # infer       : infer로부터 strategy가 진행됩니다. 
    # portfolio   : universe.csv와 weight.csv가 존재할 경우에 이 데이터부터 strategy가 진행됩니다
    "from_which": "infer",  
    
    # long포트폴리오의 총 비중입니다. ex) longonly의 경우 1.0  , 130/30의 경우 1.3,
    "long_amount": 1.0,  
    "long_picking_config": picks.picking_by_signal("out", False, 1, None, ascending=False),
    "long_weighting_config": (
        {'name': 'custom_w', 'custom_weighting_func': custom_weight},
    ),

    ########### 마스터에서 실험할때는 아래를 따로 설정해줄 필요가 없습니다 ############
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
    "use_neptune": True,  # 넵튠을 사용하려면 True로 표시합니다.
    "user_id": "jayden",  # USER_ID는 당분간 저로 고정합니다
    "project_name": "jayden", # 프로젝트는 jayden, tei, aiden으로 만들어 뒀습니다

    "exp_name": "실험의 이름을 설정합니다",  # 실험의 이름 필수는 아닙니다
    "description": "실험의 설명을 작성합니다",  # 실험의 설명 필수는 아닙니다

    "hparams": {**strategy_dict}, # 저장하고 싶은 하이퍼 파라미터를 딕셔너리로, 굳이 안넣어줘도 소스코드가 저장되어 당시의 셋팅을 확인가능합니다.

    "tags": [],  # 마스터 프로젝트에서는 태그를 변경하지 않습니다
    "exp_id": ['NEW'],  # 마스터 프로젝트에서는 EXP_ID를 변경하지 않습니다
    "token": NEPTUNE_TOKEN, # 키는 고정입니다
}

################################################################################################### 
################################################################################################### 
################################################################################################### 
setting_params = {
    "identifier": IDENTIFIER,  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "jh_master_cache").as_posix(),
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
            "name": attention_model.AttentionModel.__name__,
            "cs_dims": 10,
            "ts_dims": 30,
            "bs_dims": 10,
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
    "batch_size": 1024,
    "validation_ratio": 0.2,
    "shuffle": True,
}

def qrft_running(_, context_name_list, lc_list, shapes_dict, d_inference):
    di = DeepIntegration(
        identifier=setting_params["identifier"],
        sub_identifier=d_inference.strftime("%Y-%m-%d"),
        dataset=None,
        data_shape=shapes_dict,
        integrated_model_params=integrated_model_params,
        setting_params=setting_params,
    )

    training_lc = lc_list[0]
    inference_lc_list = lc_list

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
        for meta, data in training_lc.split_of(split_item.VALIDATION).training_iterate(
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
        for i, (context_name, lc) in enumerate(zip(context_name_list, inference_lc_list)):
            for meta, data in lc.inference_iterate():
                sec_keys = meta["SECURITY"]

                if integrated_model_params["save_all_epochs"]:
                    di.infer_with_all_epochs(None, data, sec_keys)

                if epoch == integrated_model_params["epochs"] or early_stopping_flag:
                    if i == 0:
                        di.infer_with_all_epochs(None, data, sec_keys)
                    else:
                        di.infer_with_all_epochs(f"{context_name}", data, sec_keys)
                    di.summary_after_infer(epoch, integrated_model_params["epochs"], best_loss)
            di.run_after_epoch()


if __name__ == "__main__":
    universe = Universe(**setting_params["kirin_config"])
    api = get_kirin_api(universe)
    load_dataset = LoadData(path=Path(DATA_DIR) / "jh_master_dataset", universe=universe)

    with mp.Pool(8) as pool:
       res = pool.starmap(load_dataset.call_if_not_loaded, input_data)

    sector_values = load_dataset.call_if_not_loaded(
        "sector_values",
        ["compustat", "read_sql"],
        ["SELECT giccd FROM r_giccd WHERE gictype='GSECTOR';"]
    )
    sector_values = sector_values.values.reshape(-1)
    for i, qdata in enumerate(res):
       if qdata.get_tag() == QraftDataTag.INDEX.value:
           qdata = qdata.rolling(SAMPLE_LENGTH).zscore()

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
    infer_filter_base = load_dataset.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    infer_filter_base._data = infer_filter_base._data.notna()

    infer_universes = [
        UNIVERSE1,
        UNIVERSE2,
        UNIVERSE3,
        UNIVERSE19951231,
        UNIVERSE20021231,
        UNIVERSE20101231,
        UNIVERSE20151231
    ]

    name_list = ["U1", "U2", "U3", "SU95", "SU02", "SU10", "SU15"]
    infer_filters = []
    for universe_gk in infer_universes:
        infer_filter = infer_filter_base.copy()
        difference = infer_filter._data.columns.difference(universe_gk)
        infer_filter.loc[:, difference] = False
        infer_filters.append(infer_filter)

    # Check input, output and filter dataset has same index and columns
    index, columns = bm.checking.check_equal_index_and_columns(
       input_data + [additional_mv, output, infer_filter_base]
    )

    sbm_list = []
    for name, infer_filter in zip(name_list, infer_filters):
        input_x = bm.DataBinderV3(
            data_list=input_data,
            training_filter=training_filter_mv,
            inference_filter=infer_filter,
            length=SAMPLE_LENGTH,
            is_input=True,
            max_nan_ratio=0.99,
            aug=None,
        )
        input_mv = bm.DataBinderV3(
            data_list=[additional_mv],
            training_filter=training_filter_mv,
            inference_filter=infer_filter,
            length=1,
            is_input=True,
            max_nan_ratio=0.99,
            aug=None,
        )
        output_y = bm.DataBinderV3(
            data_list=[output],
            training_filter=training_filter_mv,
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
            data_map={
                "x": input_x,
                "mv": input_mv,
                "y": output_y,
            },
            max_cache="15GB",
            probability=None,
        )
        sbm_list.append(sbm)

    dates_inference = pd.date_range(
       start=date_dict["date_from"],
       end=date_dict["date_to"],
       freq=date_dict["rebalancing_terms"],
    ).to_pydatetime()

    shapes_dict = sbm_list[0].get_sample_shapes()

    run_proc = MultiProcess()
    run_proc.cpu_count(max_count=setting_params['cpu_count'])


    for d_inference in dates_inference:
        length_list = [TRAINING_LENGTH] + [0] * (len(sbm_list) - 1)
        lc_list = [sbm.local_context(inference_dates=d_inference, length=length) for sbm, length in
                   zip(sbm_list, length_list)]
        proc = mp.spawn(qrft_running, args=(name_list, lc_list, shapes_dict, d_inference), join=False).processes[0]
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
        "UNIVERSE1_MCAP", "UNIVERSE1_DOLLAR",
        "UNIVERSE20021231_MCAP", "UNIVERSE20021231_DOLLAR",
        "UNIVERSE20101231_MCAP", "UNIVERSE20101231_DOLLAR",
        "UNIVERSE20151231_MCAP", "UNIVERSE20151231_DOLLAR",
        "UNIVERSE19951231_MCAP", "UNIVERSE19951231_DOLLAR",
        "UNIVERSE2_MCAP", "UNIVERSE2_DOLLAR",
        "UNIVERSE3_MCAP", "UNIVERSE3_DOLLAR",
    ]

    INFER_LIST = [ 'U1', 'SU02', 'SU10', 'SU15', 'SU95', 'U2', 'U3']

    for i in range(len(INFER_LIST)):
        comparison_dict["performance_fps"] = [p_pfs_list[i]]
        comparison_dict["performance_names"] = [p_names_list[i]]
        comparison_dict["standard_benchmarks"] = UNIVERSE_LIST[2 * i:2 * (i + 1)]

        neptune_dict["tags"] = [NEPTUNE_IDENTIFIER] + [INFER_LIST[i]]

        cp = Comparison(
            data_dir=DATA_DIR,
            setting_params=setting_params,
            neptune_params=neptune_dict,
            **comparison_dict,
        )
        cp.compare()






















'''
학습 모델과, weighting 방법을 스크립트 내에서 커스터마이즈 할 수 있도록 변경한 것의 예제입니다

BaseModel을 상속받아서 custom 모델을 구현한 후,
기존의 딕셔너리에서 문자열 custom이 들어가도록 'custom_model' key, value 는 커스텀모델 클래스 값으로. 전달하면 된다
나머지는 기존과 동일하다
     {
            "name": 'custom_amom',
            "custom_model" : CustomAmomModel  ,
            "fill_nan_value": int("1000000000000000000"),
            "inputs": ["x"],
            "targets": ["y"],
            "forwards": [],
        }

strategy 의 weighning 방법론 또한 딕셔너리리로 넘겨주면 된다
{
        'name': 'custom_mw',
        'weighting_function': custom_market_weight(),
    },

'''


# 커스텀 model 방법론 예시
'''
class CustomAmomModel(BaseModel):
    def __init__(self, identifier, sub_identifier,
                 model_name, input_names, target_names, forward_names,
                 input_shapes, target_shapes, adversarial_info, device, save_path,
                 fill_nan_value):
        super().__init__(identifier, sub_identifier,
                         model_name, input_names, target_names, forward_names,
                         input_shapes, target_shapes, adversarial_info, device, save_path)

        self.fill_nan_value = fill_nan_value

        self.net = SingleEquityModel(
                num_sequence=input_shapes[0],
                num_features=input_shapes[1],
                ).double().to(self.device)
        self.optimizer = get_optimizer_function(OPTIMIZER.COCOB)(self.net.parameters())
        self.dtype = self.get_dtype()

    def optimize(self, loss: torch.Tensor) -> NoReturn:
        loss.backward()
        for name, param in self.net.named_parameters():
            if name == 'lw':
                param.grad *= -1
                break
        self.optimizer.step()
        self.optimizer.zero_grad()

    def optimize_at_mini_batch(self, loss, params: list):
        if 'optimizer' in params:
            self.optimize(loss)
        if 'lr_scheduler' in params:
            self.lr_scheduler.step()

    def optimize_at_epoch(self, params: list):
        # if 'optimizer' in params:
        #     self.optimize(loss)
        if 'lr_scheduler' in params:
            self.lr_scheduler.step()

    def calculate_loss(self, data: dict, is_infer=False) -> torch.Tensor:
        if not isinstance(data, dict):
            raise TypeError("The types of x must be torch.Tensor.")

        if is_infer:
            return torch.tensor(-10000000, dtype=self.get_dtype())

        x = data['x']
        x = torch.where(torch.isfinite(x), x, torch.full_like(x, self.fill_nan_value))

        y = data['y']
        x = x.to(dtype=self.dtype, device=self.device).double()
        y = y.to(dtype=self.dtype, device=self.device).double()

        xo, xd, emb = self.net(x, emb_out=True)
        xw, xm, xvar = xo
        pdfs = self.pdf_of_normal(y, xm, xvar)
        glp = torch.log(1e-8 + torch.sum(xw*pdfs, dim=-1))
        tc = -torch.mean(glp)

        # reconstruction loss
        ae = get_loss_function(LOSSES.MSE)(xd, x, reduction='sum')
        ac = ae/xd.shape[0]

        # association loss
        abac, vc = self.continuous_association_loss(emb, y)

        w = nn.Softmax(dim=-1)(self.net.lw)
        loss = w[0]*tc + w[1]*ac + w[2]*abac + w[3]*vc
        return loss

    def calculate_step_info_with_loss(self, data: dict, is_infer=False) -> dict:
        if not isinstance(data, dict):
            raise TypeError("The types of x must be dictionary.")

        return {'loss': self.calculate_loss(data, is_infer)}

    def predicts(self, data: dict) -> Union[Tuple[str, np.ndarray], Tuple[Tuple[str, np.ndarray]]]:
        if not isinstance(data, dict):
            raise TypeError("The type of y should be dictionary.")

        self.set_validation_mode()
        x = data['x']
        x = torch.where(torch.isfinite(x), x, torch.full_like(x, self.fill_nan_value))

        x = x.to(dtype=self.dtype, device=self.device)
        x = torch.tensor(x, device=self.device).double()
        xo, _ = self.net(x)
        xw, xm, xvar = xo
        gmu = torch.sum(xw*xm, dim=-1, keepdim=False).detach().cpu().numpy()
        return 'mu', gmu

    @staticmethod
    def continuous_association_loss(emb, y):
        dmat = y - y.T
        dmat = torch.pow(dmat, 2)
        match_mat = torch.mm(emb, emb.T)
        p_ab = torch.nn.Softmax(dim=-1)(match_mat)
        p_ba = torch.nn.Softmax(dim=-1)(match_mat.T)
        p_aba = torch.mm(p_ab, p_ba)

        l_aba = torch.mean(dmat*torch.log(p_aba + 1e-8))
        visit_p = torch.sum(p_ab, dim=0, keepdim=True)
        l_visit = -torch.mean(torch.log(visit_p + 1e-8))

        return l_aba, l_visit

    @staticmethod
    def pdf_of_normal(x, mu, var):
        return torch.exp(-(x-mu)**2/(2*var)) / (2*np.pi*var)**0.5

'''
# 커스텀 weighting 방법론 예시
'''

def custom_market_weight():
    def _custom_market_weight(dt, signal, kirin_config, mv):
        w = signal.sort_values(ascending=True)
        return w

    return _custom_market_weight
'''





