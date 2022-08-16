### 하베스트: 시총상위 20% 중에 중국 기업 아닌 것 중에 탄소배출량 상위 20%는 아닌 것을 고른다 ###
import sys
import time
sys.path.append("/home/sronly/Projects/qiscripts")
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

import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from qraft_data.data import QraftData

class Harvest:
    def __init__(self, harvest_config, meta):
        """
        :param harvest_config: "emissions_fp":"./harvest/emissions_intensity_per_sales.xlsx" 형식의 xlsx 파일
        :param meta: api.compustat.security_meta_table 형태의 pandas dataframe 파일
        """
        self._harvest_config = deepcopy(harvest_config)
        self._emissions_df = self._read_emissions_file(harvest_config)
        self._tickers = set(self._emissions_df.columns)
        #self._api = api
        self._gvkey_iid_to_tic_df = meta[["gvkey_iid", "tic"]].drop_duplicates()\
                .set_index("gvkey_iid").squeeze()
        self._meta = meta.set_index("gvkey_iid")

    def _read_emissions_file(self, harvest_config):
        """
        emission xlsx 파일을 불러와, index는 date고 columns는 ticker이며 각 cell의 값은 carbon emission인 데이터프레임을 반환한다
        :param harvest_config: harvest_config: "emissions_fp":"./harvest/emissions_intensity_per_sales.xlsx" 형식의 xlsx 파일
        :return: index는 date고 columns는 ticker이며 각 cell의 값은 carbon emission인 데이터프레임
        """
        df = pd.read_excel(Path(harvest_config["emissions_fp"]))
        df = df[df.columns[:3]]
        df = df.rename(
                columns={
                    "Data Date": "date",
                    "GHG/CO2 Emissions Intensity per Sales": "emissions_per_sale",
                    "Ticker": "ticker"
                    })
        df.ticker = df.ticker.str.split().str[0]
        df['date'] = pd.to_datetime(df['date'])
        df = df.pivot_table(index='date', columns="ticker", values="emissions_per_sale")
        return df

    def _gvkey_iid_to_ticker(self, gvkey_iid):
        """
        CO2 emission이 있는 기업에 대해서만 gvkey_iid를 티커로 변환한다
        :param gvkey_iid: gvkey_iid
        :return: CO2 emission이 있는 기업에 대한 gvkey_iid를 티커로 변환한 결과
        """
        ticker = self._gvkey_iid_to_tic_df.loc[gvkey_iid]
        if isinstance(ticker, pd.Series):
            # if there are more than two tickers
            ticker_filter = ticker.apply(lambda x: x in self._tickers)
            if ticker_filter.sum() > 1:
                df = self._meta.loc[gvkey_iid]
                tickers = ticker[ticker_filter].tolist()
                df = df.loc[df['tic'].isin(tickers)]
                return df.iloc[df['effdate'].argmax()]['tic']
            if ticker_filter.sum() == 0:
                return np.nan

            return ticker[ticker_filter].item()

        return ticker if ticker in self._tickers else np.nan

    def get_china_filter(self, date_list, gvkey_iid_list):
        """
        fic 필터와  loc 필터 둘중 하나라도 중국 기업이면 제외한, 비 중국기업이 True인 mask를 만든다
        :param date_list: 월말 날짜
        :param gvkey_iid_list: gvkey_iid의 리스트
        :return: fic 필터와  loc 필터 둘중 하나라도 중국 기업이면 제외한, 비 중국기업이 True인 mask
        """
        CHINA_SYM = "CHN"
        sr = (self._meta["fic"] != CHINA_SYM) & (self._meta["loc"] != CHINA_SYM)
        sr = sr.groupby(sr.index).all()
        sr = sr.reindex(gvkey_iid_list)

        df = pd.DataFrame(index=date_list, columns=gvkey_iid_list, data=True)
        df[sr.index[sr==False]] = False
        df = QraftData("filter_china", df)
        return df

    def get_emissions_filter(self, date_list, gvkey_iid_list,
            filter_prev=None, exclusion_pct=0.2, complement=False, drop_missing=True):
        """
        탄소배출량 상위 20%가 아닌 것을 고른다
        :param date_list: 월말 날짜
        :param gvkey_iid_list: gvkey_iid의 리스트
        :param filter_prev: 사전에 정의한 다른 필터(예를 들면 시가총액 상위 30% 필터)
        :param exclusion_pct: 탄소 배출량 상위 몇 퍼센트까지 제외할 것인지에 대한 값
        :param complement: 마스크 반전에 관한 값(당장 필요는 없다)
        :param drop_missing: emission data 자체가 없는 기업을 뺄지 넣을지를 결정하는 값
        :return: 하베스트: 시총상위 20% 중에 중국 기업 아닌 것 중에 탄소배출량 상위 20%는 아닌 것을 고른다
        """
        ticker_list = gvkey_iid_list.map(self._gvkey_iid_to_ticker)
        df = pd.DataFrame(index=date_list, columns=gvkey_iid_list)
        data = self._emissions_df.reindex(date_list)
        idx_ticker_exists = np.nonzero(ticker_list.notna())[0]
        df.iloc[:, idx_ticker_exists] = data.loc[:, ticker_list.dropna()]
        df = df.ffill()

        if filter_prev is not None:
            df = df.where(filter_prev.data)

        df = df.rank(axis=1, ascending=False, pct=True)
        df = QraftData("filter_emission", df)

        if drop_missing:
            if complement:
                return df < exclusion_pct
            return df >= exclusion_pct
        else:
            if complement:
                if filter_prev is not None:
                    return QraftData("filter_emission",
                            ~(df.data >= exclusion_pct) & filter_prev.data)
                return QraftData("filter_emission",
                    ~(df.data >= exclusion_pct))
            if filter_prev is not None:
                return QraftData("filter_emission",
                        ~(df.data < exclusion_pct) & filter_prev.data)
            return QraftData("filter_emission",
                    ~(df.data < exclusion_pct))



#################################### 공통설정부분 ##########################################
setting_params = {
    "identifier": "Harvest_Green_concept001_exp01_test02",  # 실험의 이름입니다
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

    "cpu_count": 2
}

date_dict = {
    "date_from": "2016-10-31",
    "date_to": "2021-10-31",
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
    "epochs": 10,
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

neptune_dict = {
    "use_neptune": False,  # 테스트기간동안에는 잠시 False로 해두겠습니다.
    "user_id": "qrft",
    "project_name": "qrft",
    "exp_id": "qrft-0228",
    "exp_name": "qrft",
    "description": "qrft",
    "tags": ["qrft"],
    "hparams": {},
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
        di.run_after_epoch()


if __name__ == "__main__":
    _back_length = 72
    _input_length = 36
    universe = Universe(**setting_params["kirin_config"])
    api = get_kirin_api(universe)

    load_dataset = LoadData(path=Path(DATA_DIR)/'kirin_dataset', universe=universe)

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

    # Masking training & inference filter
    filter_mv = load_dataset.call_if_not_loaded("mv", ["high_level", "equity", "get_monthly_market_value"])
    #filter_mv = filter_mv.rank(350=False, pct=True) <= 0.15
    filter_mv = filter_mv.rank(ascending=False, pct=True) <= 0.2

    # Masking inference for Harvest
    meta = api.compustat.security_meta_table
    harvest = Harvest(
            {"emissions_fp":"./harvest/emissions_intensity_per_sales.xlsx"}, meta)


    # Masking inference for Harvest
    # meta = api.compustat.set_investment_universe(**universe.pre_filters)

    filter_china = harvest.get_china_filter(filter_mv.index, filter_mv.columns)

    filter_prev = QraftData("filter_mv_china", filter_mv.data & filter_china.data)

    filter_emission = harvest.get_emissions_filter(
        filter_mv.index, filter_mv.columns,
        filter_prev=filter_prev, drop_missing=True)

    infer_filter = filter_emission  # if no exlusion filter_prev


# Check input, output and filter dataset has same index and columns
    index, columns = bm.checking.check_equal_index_and_columns(
       input_data + [additional_mv, output, filter_mv]
    )

    input_x = bm.DataBinder(
       data_list=input_data,
       training_filter=filter_mv,
       inference_filter=infer_filter,
       length=_input_length,
       is_input=True,
       max_nan_ratio=0.99,
       aug=None,
    )
    input_mv = bm.DataBinder(
       data_list=[additional_mv],
       training_filter=filter_mv,
       inference_filter=infer_filter,
       length=1,
       is_input=True,
       max_nan_ratio=0.99,
       aug=None,
    )
    output_y = bm.DataBinder(
       data_list=[output],
       training_filter=filter_mv,
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
        identifier=setting_params['identifier'],
        kirin_config=setting_params['kirin_config'],
        date_from=date_dict['date_from'],
        date_to=date_dict['date_to'],
        rebalancing_terms=date_dict['rebalancing_terms'],

        long_picking_config=picks.picking_by_signal("out", False, 1, 88, ascending=False),
        long_weighting_config=(
            weights.rank_sum_discounted_mix_with_top_market_weight(0.995, 70, 350, 1.0),
            weights.optimal_weight(
                kirin_config=setting_params['kirin_config'],
                loss_type="MSE",
                max_weight=0.08,
                threshold_weight=0.05,
                bound_sum_threshold_weight=0.4,
                bound_gics={"sector": 0.5, "industry": 0.24},
                bound_financials_sector={"40": 0.048},
            ),
        ),

        factor=False,
        market_percentile=0.2,

        gics=False,
        gics_level=['sector'],
    )
    st.backtest()

    cp = Comparison(
        data_dir=DATA_DIR,
        identifier=setting_params['identifier'],
        kirin_config=setting_params['kirin_config'],
        performance_fps=list(st.get_performance_fps())+['Harvest_original'],
        performance_names=list(st.get_performance_names())+['Harvest_original'],
        standard_benchmarks=["S&PCOMP"],
        comparison_periods=[],
        final_score=final_scores.annualized_return(exponential_decay_rate=None, total=False),
        neptune_params=neptune_dict
    )
    cp.compare()
