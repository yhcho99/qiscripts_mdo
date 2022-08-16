# Kirin api 불러오고 설정한 후 필요한 데이터 호출하는 부분
import pandas as pd
import numpy as np
import sys
sys.path.append("/home/sronly/Projects/qiscripts")
from kirin import Kirin
from strategy_simulation.helper import weights
import strategy_simulation.strategy.weighting as weighting_collection
from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from qraft_data.data import QraftData
from copy import deepcopy
from pathlib import Path
from paths import DATA_DIR, NEPTUNE_TOKEN
import functools
import os

setting_params = {
    "identifier": "Harvest_nonAI",  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": ("/raid/sr-storage/kirin_cache"),
        "use_sub_server": False,
        "exchange": ["NASDAQ","NYSE"],
        "security_type": ["COMMON"],
        "backtest_mode": True,
        "except_no_isin_code": False,
        "class_a_only": True,
        "primary_issue": True,
        "pretend_monthend": False,
    },
}


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
            filter_prev=None, exclusion_pct=0.2, complement=False, drop_missing=False):
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

long_weighting_config = (
        weights.optimal_weight(
            kirin_config=setting_params['kirin_config'],
            loss_type="MSE",
            max_weight=0.08,
            threshold_weight=0.05,
            bound_sum_threshold_weight=0.4,
            bound_gics={"sector": 0.5, "industry": 0.24},
            bound_financials_sector={"40": 0.048},
        ), # 쉼표가 몹시 중요하다! 안쓰면 에러난다.
)


def get_api(kirin_config):
    """
    주어진 universe에서 kirin api를 생성하여 리턴한다.
    :param kirin_config:
    :return: api
    """
    universe = Universe(**kirin_config)
    api = get_kirin_api(universe)

    return api


def mv_masking(api, percent=None, select_num=None):
    """
    유니버스에서 시가총액 기준으로 상위 N%나 상위 N개를 추려낸다
    """
    MV = api.compustat.get_monthly_market_value()
    if percent is not None:
        mv_mask = MV.rank(axis=1, ascending=False, pct=True) <= percent
        mv_mask = QraftData("market_value",mv_mask)
        return mv_mask

    elif select_num is not None:
        mv_mask = MV.rank(axis=1, ascending=False, pct=False) <= select_num
        mv_mask = QraftData("market_value", mv_mask)

        return mv_mask

    return QraftData("market_value", MV)

# QRAFT Filter 만들어주는 함수
def qraft_masking(api, filter_mv):
    """
    universe 안에서 Qraft의 컨셉 필터를 만든다. 지금은 Harvest 필터이다.
    :param api: kirin api
    :param filter_mv: 시가총액 필터
    :return: 주어진 universe안에서 Qraft 컨셉 필터의 결과대로 정렬된 종목들
    """

    meta = api.compustat.security_meta_table
    harvest = Harvest(
        {"emissions_fp": "./harvest/emissions_intensity_per_sales.xlsx"}, meta)

    # Masking inference for Harvest
    # meta = api.compustat.set_investment_universe(**universe.pre_filters)

    filter_china = harvest.get_china_filter(filter_mv.index, filter_mv.columns)

    # filter_prev = QraftData("filter_mv_china", filter_mv.data & filter_china.data)
    filter_prev = QraftData("filter_china", filter_china.data)

    filter_emission = harvest.get_emissions_filter(
        filter_mv.index, filter_mv.columns,
        filter_prev=filter_prev, drop_missing=False)

    infer_filter = filter_emission

    return infer_filter

# 최적 가중치 찾아주는 상위 부분
def get_weighting_function(weighting_config):
    """
    제약 조건 하 weighting 방법론을 리턴한다
    :param weighting_config: weighting 방법론(현재는 long_weighting_config)
    :return: weighting 방법론
    """
    weighting_funcs = []
    for cfg in weighting_config:
        name = cfg.pop('name')
        func = getattr(weighting_collection, name)(**cfg)
        weighting_funcs.append(func)

    return weighting_funcs


def apply_weighting_funcs(map_of_date_to_picks,
                          date,
                          weighting_funcs,
                          market_value,
                          trade_volume,
                          ):
    """
    weighting 방법론(제약조건) 하에서 실제 포트폴리오를 산출한다.
    :param map_of_date_to_picks: pandas DataFrame의 형태이며, gvkey_iid와 그에 짝지어진 시가총액 비중으로 이루어져 있다
    :param date: 뽑고 싶은 날짜
    :param weighting_funcs: weighting 제약조건
    :param market_value: 시가총액
    :param trade_volume: 거래량
    :return: 조절된 비중하의 포트폴리오
    """
    kirin_config = setting_params['kirin_config']
    key_score_df = map_of_date_to_picks.loc[date]
    long_score_df = key_score_df.dropna()
    w = functools.reduce(lambda x, f: f(date, x, kirin_config, market_value, trade_volume),
                         [long_score_df] + weighting_funcs[0])

    w = w.sort_values(ascending=False)
    return w


def get_port_return(port_weight,datetime):
    """
    포트폴리오의 수익률을 산출한다
    :param port_weight: 조절된 가중치 하의 포트폴리오
    :return: price return 기준 수익률
    """
    TR = api.compustat.get_monthly_total_return(1, 0)
    port_tr = (port_weight.shift(1) * TR).sum(1)
    port_tr = port_tr.iloc[-len(datetime)+1:]

    return port_tr


def get_portfolio(mask: pd.DataFrame, port_weight: pd.DataFrame, datetime, select):
    """
    gvkey_iid로 되어있는 포트폴리오를 ticker_weight 형식으로 바꾼다
    :param api: kirin api
    :param mask: 포트폴리오 구성(gvkey_iid)
    :param port_weight: 비중 조절된 포트폴리오 가중치
    :param datetime: 날짜
    :param select: 최종 포트폴리오 수
    :return: ticker_weight이 각 cell인 pandas DataFrame
    """

    length_of_monthIndex = mask.shape[0]
    np_mask = np.array(mask)
    universe_df = pd.DataFrame(index=mask.index, columns=range(select))
    weight_df = pd.DataFrame(index=port_weight.index, columns=range(select))
    for i in range(length_of_monthIndex-len(datetime), length_of_monthIndex):
        print(mask.index[i])
        dict = {}
        selected_company_indexList = np.where(np_mask[i, :] == True)[0].tolist()
        for selected_company_index in selected_company_indexList:
            company_gvkey_iid = mask.columns[selected_company_index]
            weight = port_weight.loc[mask.index[i], company_gvkey_iid]
            dict[company_gvkey_iid] = round(float(weight), 4)

        sorteddict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        company = []
        percent = []
        for val in sorteddict:
            company.append(val[0])
            percent.append(val[1])
        for j in range(len(company)):
            universe_df.iloc[i,j] = str(company[j])
            weight_df.iloc[i,j] = str(percent[j])

    universe_df = universe_df.iloc[-len(datetime):,:]
    weight_df = weight_df.iloc[-len(datetime):,:]
    return universe_df, weight_df

if __name__ == "__main__":
    api = Kirin()
    api.compustat.set_investment_universe(
        exchange=["NASDAQ", "NYSE"],
        security_type=["COMMON"],
        class_A_only=True,
    )
    meta = api.compustat.security_meta_table
    select = 88
    mv_mask = mv_masking(api, percent=None, select_num=None) #Market Value
    long_weighting_funcs = get_weighting_function(weighting_config=long_weighting_config)
    weighting_func = [long_weighting_funcs]

    qraft_concept_mask = qraft_masking(api, mv_mask) #Harvest

    mv_qraft_concept_mask = mv_masking(api,percent=None,select_num=None).masked_by(qraft_concept_mask)
    mv_qraft_concept_selected_mask = mv_qraft_concept_mask.rank(ascending=False,pct=False) <= select
    mv_qraft_concept_weight = mv_qraft_concept_mask.masked_by(mv_qraft_concept_selected_mask)
    port_weight = mv_qraft_concept_weight.div(mv_qraft_concept_weight.sum(axis='columns'), axis='index')

    dates = pd.date_range("2015-12-31", "2021-10-31", freq='M').strftime('%Y-%m-%d').to_list()

    changed_port_weight = pd.DataFrame(index=port_weight.index, columns=port_weight.columns)

    market_value = api.high_level.equity.get_monthly_market_value()
    trade_volume = api.compustat.get_monthly_price_data(adjust_for_split=False, adjust_for_total_return=False) \
                   * api.compustat.get_monthly_volume_data()

    for dt in dates:
        print(dt)
        w = apply_weighting_funcs(port_weight, dt, weighting_func, market_value, trade_volume)
        for gvkey_iid in w.index:
            changed_port_weight.loc[dt, gvkey_iid] = w.loc[gvkey_iid]

    port_pr = get_port_return(changed_port_weight,dates)

    # portfolio = get_portfolio_constitution(api, mv_qraft_concept_selected_mask, changed_port_weight, dates)
    # portfolio.to_csv("/raid/sr-storage/Harvest_nonAI/" + str(select) + "_portfolio.csv")

    universe, weight = get_portfolio(mv_qraft_concept_selected_mask, changed_port_weight, dates, select)
    if not os.path.exists(f"/raid/sr-storage/Harvest_nonAI/{select}"):
        os.makedirs(f"/raid/sr-storage/Harvest_nonAI/{select}")
    port_pr.to_csv(f"/raid/sr-storage/Harvest_nonAI/{select}/{select}_total_return.csv")
    universe.to_csv(f"/raid/sr-storage/Harvest_nonAI/{select}/{select}_universe.csv")
    weight.to_csv(f"/raid/sr-storage/Harvest_nonAI/{select}/{select}_weight.csv")