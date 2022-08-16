
import strategy_simulation.helper as helper
from strategy_simulation.comparison import Comparison
from attrdict import AttrDict


# test1
# def _is_keys_alive_at(self, keys: List, dt: Any)->List[bool]: 의 출력형식 확인하기

# test2
# weight_adjusting_unitdate가 실행됐을때와 그냥 1개월마다 리밸런싱하는 것으로 되었을때 map_of_dates_to_picks가 어떻게 바뀌었는지

# test3      b
# def _get_weight_as_goes(self, next_keys:List[str], next_dt: str, w: pd.Series)-> pd.Series: w의 타입 확인하기

# test4
# gics level list값으로도 동작하나 확인하기

setting_params = {
    'identifier': 'nvq_deep_350',
    'kirin_config': {
        'cache_dir': 'kirin_cache',
        'use_sub_server': False,
        'exchange': ['NYSE', 'NASDAQ'],
        'security_type': ['COMMON'],
        'except_no_isin_code': False,
        'class_a_only': True,
        'pretend_monthend': False
    },
}

comparison_dict = {
    'performance_fps': [], # identifier와 동일한 값 혹은, 전체 performance file paths
    'performance_names': [],
    'standard_benchmarks': ['U:SPY'],
    'comparison_periods': [],
    'factor': False,
    'market_cap': 0.3,
    'gics': False,
    'gics_level': 'industry',
    'final_score': helper.final_scores.annualized_return(exponential_decay_rate=None, total=False),
}

neptune_setting = {
    'use_neptune': False,
    'user_id': 'jayden',
    'project_name': 'nvq',
    'exp_id': 'NVQ-14',
    'exp_name': 'hi',
    'description': 'This experiments are the most important',
    'tags': ['nvq', 'attention'],
    'hparams': { **comparison_dict},
    'token': 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiN2VlMDM5ZGItMWVjZi00N2Q2LTk3N2EtMDlhN2VjMGI1YjdjIn0=',
}

neptune_params = AttrDict(neptune_setting)

_cp = Comparison(setting_params=setting_params, neptune_params = neptune_params, **comparison_dict)
_cp.compare()