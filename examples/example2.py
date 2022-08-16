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
        'use_sub_server': False,
        'exchange': ['NYSE', 'NASDAQ'],
        'security_type': ['COMMON'],
        "start_date": "2019-10-31",
        "end_date": "2020-03-31",
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
    'rebalancing_terms': '2M',
}


strategy_dict = {
    'short': False,
    'long_amount': 1.0,
    'short_amount': 0.0,
    'long_picking_config': picks.picking_by_signal('mu/variance**0.5', False, 1, 100, ascending=False),
    'short_picking_config': picks.picking_by_signal('out', False, 1, 30, ascending=True),
    'long_weighting_config': (
        weights.market_weight(),
        weights.optimal_weight(
            kirin_config=setting_params['kirin_config'],
            loss_type='MSE',
            max_weight=0.08,
            threshold_weight=0.05,
            bound_sum_threshold_weight=0.4,
            factor_market_percentile=0.2,
            bound_factors=1.15,
            bound_gics={
                'sector': 0.5,
                'industry': 0.24
            }
        )
    ),
    'short_weighting_config': weights.market_weight(),
    'weight_adjusting_unitdate': False,
    'backtest_daily_out': False,
    'backtest_daily_out_lag': [0, 1],

    ########### uw로 구할때는 위에 것들은 따로 설정해줄 필요가 없습니다 ############
    'from_which': 'portfolio',  #  portfolio : universe.csv와 weight.csv가 존재할 경우입니다

    'factor': False, # factor 와 관련된 백테스트를 할지 여부
    'save_factor': True, # factor와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    'market_percentile': 0.3, # 시가총액 상위 몇 %의 주식을 볼지

    'gics': False, # gics 와 관련된 백테스트를 할지 여부
    'save_gics': True, # gics와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    'gics_level': ['industry', 'sector'] # gics 레벨 결정
}


comparison_dict = {
    'performance_fps': [],
    'performance_names': [],
    'standard_benchmarks': [],
    'comparison_periods': [],
    'final_score': final_scores.annualized_return(exponential_decay_rate=None, total=False),
}
NEPTUNE_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiN2VlMDM5ZGItMWVjZi00N2Q2LTk3N2EtMDlhN2VjMGI1YjdjIn0='

neptune_dict = {
            "use_neptune": True,  # 테스트기간동안에는 잠시 False로 해두겠습니다.
            "user_id": "jayden", # 자신의 ID입니다 한 번만 설정해주면 됩니다
            "project_name": "master", # 프로젝트는 웹에서 한 번만 설정해주면 됩니다

            "exp_id": ['NEW'], # 기존에 실험의 값을 추가하거나 변경하고 싶다면 해당 실험id를(ex MAS-01 넣어주고 아니면 NEW라고 적어놓습니다
            "exp_name": "실험의 이름을 설정합니다", #필수는 아님
            "description": "실험의 설명을 작성합니다", #필수는 아님
            "tags": ["master","qrft"], #필수는 아님
            "hparams": { # 딕셔너
                **comparison_dict,
                'identifier': 'test'
            },
            "token": NEPTUNE_TOKEN,
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