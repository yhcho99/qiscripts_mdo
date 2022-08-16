from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores

from pathlib import Path
from attrdict import AttrDict

# 향후 있을 실험의 다양성을 위해 자유도를 높였습니다 불편하거나 필요없다고 생각되시는 부분이 있다면 언제든 이슈 남겨주세
# 넵튠 사용에 대해서는 https://github.com/qraft-technologies/neptune-logger 를 참고해주세요


#데이터가 저장되는 경로 설정을 해줍니다
DATA_DIR = '/home/jaehoon/QRAFT/data'

#실험이름과 kirin설정을 해줍니다.
#   - identifier는 기존처럼 폴더명과 동일하게 설정이 됩니다.


setting_params = {
    'identifier': 'qrft_part',  # 1. 실험의 이름을 설정합니
    'kirin_config': {
        'cache_dir': (Path(DATA_DIR) / 'kirin_cache').as_posix(),
        'use_sub_server': False,
        'exchange': ['NYSE', 'NASDAQ'],
        'security_type': ['COMMON'],
        "end_date": "2020-12-31",
        "backtest_mode": False,
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
    'date_from': '2019-12-31',
    'date_to': '2021-01-25',
    'rebalancing_terms': 'M',
}


strategy_dict = {
    # short을 할지의 여부입니다
    'short': False,
    # 롱숏 별 비중
    'long_amount': 1.0,
    'short_amount': 0.0,
    # 롱숏 별 picking 방법
    'long_picking_config': picks.picking_by_signal('out', False, 1, 350, ascending=False),
    'short_picking_config': picks.picking_by_signal('out', False, 1, 30, ascending=True),
    # 롱숏 별 weighting 방버
    'long_weighting_config': (weights.test_weight(0.995, 350, 0.7),
                              weights.optimal_weight(
                                  kirin_config=setting_params['kirin_config'],
                                  loss_type='MSE',
                                  max_weight=0.08,
                                  threshold_weight=0.05,
                                  bound_sum_threshold_weight=0.4,
                                  factor_market_percentile=0.2,
                                  bound_factors=1.15,
                                  bound_gics={'sector': 0.5, 'industry': 0.24},
                              )),
    'short_weighting_config': weights.market_weight(),

    'weight_adjusting_unitdate': False,

    ########### 포트폴리오에서로 구할때는 위에 것들은 따로 설정해줄 필요가 없습니다 ############
    'backtest_daily_out': False,      # 3. 월별로 구성된 포트폴리오를 일별로 확장하여 백테스트를 할 것인지 여부를 True로 해줘야합니다
    'backtest_daily_out_lag': [0, 1], # 3. custom basket이슈로 lag를 주고 싶다면 List[int]값으로 줍니다

    'from_which': 'portfolio',  # 2. from_which == 'portfolio' : universe.csv와 weight.csv가 존재할 경우만을 대상으로 합니다

    'factor': False,               # 4. factor 와 관련된 백테스트를 할지 여부 True
    'save_factor': True,          # 4. factor와 관련된 정보를 csv 파일로 저장하면 읽어서 그림을 그림
    'market_percentile': 0.3,     # 4. 시가총액 상위 몇 %의 주식을 볼지를 정해줌
    't_t1': False,                # 4. True일 경우 t시점과 t_1시점의 Factor score, False일 경우 t시점과 t시점의 Factor Score로 계산
    # 현재는 선택으로 되어있는데 선택의 경우 두번돌려야 하는 문제가 있으므로 그냥 둘다 계산하게 바꿀까요?

    'gics': True,                           # 5. gics 와 관련된 백테스트를 할지 여부 True
    'save_gics': True,                      # 5. gics와 관련된 정보를 csv 파일로 저장하면 읽어서 그림을 그림
    'gics_level': ['industry', 'sector']    # 5. gics 레벨 결정해 줌
}


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
    'token': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiN2VlMDM5ZGItMWVjZi00N2Q2LTk3N2EtMDlhN2VjMGI1YjdjIn0',
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