import strategy_simulation.helper as helper
from strategy_simulation.strategy import Strategy

# test1
# weight_adjusting_unitdate가 실행됐을때와 그냥 1개월마다 리밸런싱하는 것으로 되었을때 map_of_dates_to_picks가 어떻게 바뀌었는지

# test2
# 롱숏이 infer 점수에 따라 잘 뽑히는 지 확인하기
# 롱숏 비중이 잘 부여되는지 확인하기
# 롱숏 비중 합이 1인지 확인하기

# test3
# daily의 경우도 잘 동작하는지 확인하

# test4
# uw를 읽어들여서 시작한 performance파일과
# 원래 infer부터 시작한 performance file이 동일한 값을 출력하는지 확인하기


from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores

from pathlib import Path
from attrdict import AttrDict

'''
1. 예시파일의 경우 데이터가 저장되는 경로를 설정해 줍니다/ qiscripts에서는 경로는 자동으로 설정됩니다
2. identifier를 포트폴리오 universe weights가 저장된 곳으로 설정해 줍니다
3. strategy_dict의 from_which아래에 있는 내용들을 목적에 맞게 수정합니다 
'''


#데이터가 저장되는 경로 설정을 해줍니다
DATA_DIR = '/Users/jaehoon/QRAFT/data'


setting_params = {
    'identifier': 'amom', #  uw가 존재하는 identifier로 설정해줍
    'kirin_config': {
        'cache_dir': (Path(DATA_DIR) / 'kirin_cache').as_posix(),
        'use_sub_server': True,
        'exchange': ['NYSE', 'NASDAQ'],
        'security_type': ['COMMON'],
        "start_date": "2020-01-31",
        "end_date": "2021-03-31",
        "backtest_mode": False,
        'except_no_isin_code': False,
        'class_a_only': True,
        'pretend_monthend': False,
    },
}

# 포트폴리오에서 읽어들일 시 date_dict는 필요 없습니
date_dict = {
    'date_from': '2020-10-31',
    'date_to': '2021-03-31',
    'rebalancing_terms': '1M',
}

def custom_market_weight():
    def _custom_market_weight(dt, signal, kirin_config, mv):
        w = signal.sort_values(ascending=True)
        return w

    return _custom_market_weight


strategy_dict = {
    'from_which': 'infer',  # portfolio : universe.csv와 weight.csv가 존재할 경우입니다

    'short': False,
    'long_amount': 1.0,
    'short_amount': 0.0,
    "long_picking_config": picks.picking_by_signal("mu", False, 1, 50, ascending=False),
    "short_picking_config": picks.picking_by_signal("out", False, 1, 30, ascending=True),
    "long_weighting_config": weights.market_weight(),
    'short_weighting_config': weights.market_weight(),
    'weight_adjusting_unitdate': False,

    'backtest_daily_out': False,
    'backtest_daily_out_lag': [0, 1],

    'factor': False, # factor 와 관련된 백테스트를 할지 여부
    'save_factor': True, # factor와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    'market_percentile': 0.2, # 시가총액 상위 몇 %의 주식을 볼지

    'gics': False, # gics 와 관련된 백테스트를 할지 여부
    'save_gics': True, # gics와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    'gics_level': ['industry', 'sector'] # gics 레벨 결정
}


comparison_dict = {
    'performance_fps': [],
    'performance_names': [],
    'standard_benchmarks': ['U:SPY'],
    'comparison_periods': [],
    'final_score': final_scores.annualized_return(exponential_decay_rate=None, total=False),
}

neptune_dict = {
    'use_neptune': False,  # 테스트기간동안에는 잠시 False로 해두겠습니다.
    'user_id': 'jayden',
    'project_name': 'nvq',
    'exp_id': 'NVQ-14',
    'exp_name': 'hi',
    'description': 'This experiments are the most important',
    'tags': ['nvq', 'attention'],
    'hparams': {**comparison_dict},
    'token': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiN2VlMDM5ZGItMWVjZi00N2Q2LTk3N2EtMDlhN2VjMGI1YjdjIn0=',
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
    neptune_params=neptune_dict,
    **comparison_dict,
)
cp.compare()