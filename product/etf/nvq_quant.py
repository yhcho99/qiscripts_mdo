import sys
sys.path.append('.')
import pandas as pd
import batch_maker as bm
import multiprocessing as mp

from pathlib import Path

from qraft_data.util import get_kirin_api
from qraft_data.universe import Universe
from qraft_data.data import Tag as QraftDataTag
from paths import DATA_DIR, NEPTUNE_TOKEN

from strategy_integration.load_data import LoadData
from strategy_integration.components.integration.rule.rule_integration import RuleIntegration
from strategy_integration.components.models.rule_models import WeightedSumModel

from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores

setting_params = {
    'identifier': 'nvq_refac',  # 실험의 이름입니다
    'kirin_config': {
        'cache_dir': (Path(DATA_DIR) / 'kirin_cache').as_posix(),
        'use_sub_server': False,
        'exchange': ['NYSE', 'NASDAQ'],
        'security_type': ['COMMON'],
        "backtest_mode": True,
        'except_no_isin_code': False,
        'class_a_only': True,
        'pretend_monthend': False,
        },

    "seed_config": {
            "training_model": True,
            "training_data_loader": None,
            "train_valid_data_split": None,
        },
    "csv_summary_save": True,
    "omniboard_summary_save": False,
    "tensorboard_summary_save": True,

}


date_dict = {
    'date_from': '2012-06-30',
    'date_to': '2020-12-31',  #
    'rebalancing_terms': 'M',
}

#################################### 배치메이커 ##########################################

universe = Universe(
    **setting_params['kirin_config']
)

# mdh = MacroDataHandler(universe).pipeline()

api = get_kirin_api(universe)
comp_api = api.compustat
custom_comp_api = api.compustat.custom_api
high_equity = api.high_level.equity
high_macro = api.high_level.macro
high_index = api.high_level.index

high_equity.get_index_for_book_to_market()
_input_length = 1
input_data = [
    ("index_btm", ['high_level', 'equity', 'get_index_for_book_to_market'], [], {}),
    ("index_ia_ta", ['high_level', 'equity', 'index_for_linear_assumed_intangible_asset_to_total_asset'], [], {}),
]

load_dataset = LoadData(path=DATA_DIR, universe=universe)

with mp.Pool() as pool:
    res = pool.starmap(load_dataset.call_if_not_loaded, input_data)

additional_mv = load_dataset.call_if_not_loaded('mv', ['high_level', 'equity', 'get_monthly_market_value'])
filter_mv = additional_mv.rank(ascending=False, pct=True) <= 0.35

output = load_dataset.call_if_not_loaded('pr_1m_0m',  ['high_level', 'equity', 'get_monthly_price_return'], [1, 0]).masked_by(filter_mv)
additional_mv._data, output._data = additional_mv._data.loc[date_dict['date_from']:date_dict['date_to']], output._data.loc[date_dict['date_from']:date_dict['date_to']]



for i, qdata in enumerate(res):
    qdata._data = qdata._data.loc[date_dict['date_from']:date_dict['date_to']]
    qdata = qdata.masked_by(filter_mv)
    if qdata.get_tag() == QraftDataTag.EQUITY.value:
        qdata = qdata.rank(ascending=True, pct=True)

    input_data[i] = qdata

index, columns = bm.checking.check_equal_index_and_columns(input_data + [additional_mv, output])

input_x = bm.DataBinder(
    data_list=input_data,
    training_filter=filter_mv,
    inference_filter=filter_mv,
    length=_input_length,
    is_input=True,
    max_nan_ratio=0.0,
    aug=None,
)
output_y = bm.DataBinder(
    data_list=[output],
    training_filter=filter_mv,
    inference_filter=filter_mv,
    length=1,
    is_input=False,
    max_nan_ratio=0.0,
    aug=None,
)
sbm = bm.SecurityBatchMaker(
    save_path=None,
    index=index,
    columns=columns,
    data_map={
        "x": input_x,
        "y": output_y,
    },
    max_cache="15GB",
    probability=None,
)


integrated_model_params = {
    'models':
        {'name': WeightedSumModel.__name__,
         'weights': (0.7, 0.3)},
    'path': DATA_DIR,

}

dates_inference = pd.date_range(start=date_dict['date_from'],
                                end=date_dict['date_to'],
                                freq=date_dict['rebalancing_terms']).to_pydatetime()

for d_inference in dates_inference:
    with sbm.local_context(
        inference_dates=[d_inference],
        length=0
    ) as lc:
        print (d_inference)
        ri = RuleIntegration(
            identifier=setting_params['identifier'],
            sub_identifier=d_inference.strftime("%Y-%m-%d"),
            dataset=lc,
            batch_maker=sbm,
            integrated_model_params=integrated_model_params,
            setting_params=setting_params,
        )

    for meta, data in lc.inference_iterate():
        sec_keys = meta["SECURITY"]

        ri.infer_with_intermediate(data, sec_keys)

#################################### simulation ##########################################

strategy_dict = {
    'short': False,  # short을 할지의 여부입니다

    'long_amount': 1.0,  # long포트폴리오의 총 비중입니다. ex) longonly의 경우 1.0  , 130/30의 경우 1.3
    'short_amount': 0.0,  # short포트폴리오의 총 비중입니다. ex) longonly의 경우 0. , 130/30의 경우 -0.3

    'long_picking_config': picks.picking_by_signal('mu', False, 1, 100, ascending=False), # 롱포트폴리오 뽑는방법
    'short_picking_config': picks.picking_by_signal('mu', False, 1, 30, ascending=True),  # 숏포트폴리오 뽑는방법

    'long_weighting_config': (
        weights.market_weight(),
        weights.distribute_of_over_weight(0.08)
    ),
    'short_weighting_config': weights.market_weight(),  # 숏 종목 비중 주는 방법

    'weight_adjusting_unitdate': False,  # 리밸런싱 시점에 관계없이 매 시점마다 weight 구하는 방법입니다
    'backtest_daily_out': False,  # 월별로 구성된 포트폴리오를 일별로 확장하여 백테스트를 할 것인지 여부
    'backtest_daily_out_lag': [0, 1],  #

    ########### 포트폴리오에서로 구할때는 위에 것들은 따로 설정해줄 필요가 없습니다 ############
    'from_which': 'infer',  # infer: infer 데이터로부터 strategy가 진행됩니다. portfolio : universe.csv와 weight.csv가 존재할 경우입니다

    'factor': False, # factor 와 관련된 백테스트를 할지 여부
    'save_factor': True, # factor와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    'market_percentile': 0.3, # 시가총액 상위 몇 %의 주식을 볼지

    'gics': False, # gics 와 관련된 백테스트를 할지 여부
    'save_gics': True, # gics와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    'gics_level': ['industry', 'sector'] # gics 레벨 결정
}

# performance_fps 와 performance_names의 경우 strategy를 먼저 실행했을 경우
# get_performance_fps() , get_performance_names() 함수를 통해 얻을 수 있고
# comparison부터 시작할 경우엔 fps의 경우 폴더명이나 file_path 를 적어주면 됩니다

comparison_dict = {
    'performance_fps': [],  # identifier와 동일한 값 혹은, 전체 performance file paths
    'performance_names': [],  # 각 퍼포먼스 별 별칭
    'standard_benchmarks': ['U:SPY'],  # 벤치마크로 삼을 U:SPY
    'comparison_periods': [],  # 비교하고 싶은 기간
    'final_score': final_scores.annualized_return(exponential_decay_rate=None, total=False),
}

neptune_dict = {
    'use_neptune': False, # 테스트기간동안에는 잠시 False로 해두겠습니다.
    'user_id': 'jayden',
    'project_name': 'nvq',
    'exp_id': 'NVQ-14',
    'exp_name': 'hi',
    'description': 'This experiments are the most important',
    'tags': ['nvq', 'attention'],
    'hparams': { **comparison_dict},
    'token': NEPTUNE_TOKEN,
}

st = Strategy(
    data_dir=DATA_DIR,
    setting_params=setting_params,
    date_params=date_dict,
    **strategy_dict,
)
st.backtest()

comparison_dict['performance_fps'] = st.get_performance_fps()
comparison_dict['performance_names'] = st.get_performance_names()

cp = Comparison(
    data_dir=DATA_DIR,
    setting_params=setting_params,
    neptune_params = neptune_dict,
    **comparison_dict,
)
cp.compare()

