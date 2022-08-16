from strategy_simulation.strategy import Strategy
from strategy_simulation.comparison import Comparison
from strategy_simulation.helper import picks, weights, final_scores
from paths import DATA_DIR
from pathlib import Path
from attrdict import AttrDict
'''
1. 예시파일의 경우 데이터가 저장되는 경로를 설정해 줍니다/ qiscripts에서는 경로는 자동으로 설정됩니다
2. identifier를 포트폴리오 universe weights가 저장된 곳으로 설정해 줍니다
3. strategy_dict의 from_which아래에 있는 내용들을 목적에 맞게 수정합니다 
'''


setting_params = {
    "identifier": "nvq_0531",  # 실험의 이름입니다
    "kirin_config": {
        "cache_dir": (Path(DATA_DIR) / "kirin_cache").as_posix(),
        "use_sub_server": True,
        "exchange": ["NYSE", "NASDAQ"],
        "security_type": ["COMMON"],
        "backtest_mode": False,
        "except_no_isin_code": False,
        "class_a_only": True,
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

    "cpu_count": 24
}

date_dict = {
    "date_from": "2020-12-31",
    "date_to": "2021-05-31",
    "rebalancing_terms": "M",
}


strategy_dict = {
    "short": False,  # short을 할지의 여부입니다
    "long_amount": 1.0,  # long포트폴리오의 총 비중입니다. ex) longonly의 경우 1.0  , 130/30의 경우 1.3
    "short_amount": 0.0,  # short포트폴리오의 총 비중입니다. ex) longonly의 경우 0. , 130/30의 경우 -0.3
    "long_picking_config": picks.picking_by_signal("out", False, 1, 100, ascending=False),  # 롱포트폴리오 뽑는방법
    "short_picking_config": picks.picking_by_signal("out", False, 1, 30, ascending=True),  # 숏포트폴리오 뽑는방법
    "long_weighting_config": (
        weights.rank_sum_discounted_weight(0.995, 100, 0.2),
        weights.optimal_weight(
            kirin_config=setting_params["kirin_config"],
            loss_type="MSE",
            max_weight=0.08,
            threshold_weight=0.05,
            bound_sum_threshold_weight=0.4,
            bound_gics={"sector": 0.5, "industry": 0.24},
            bound_financials_sector= { "40" : 0.048 }
        )
    ),
    "short_weighting_config": weights.market_weight(),  # 숏 종목 비중 주는 방법
    "weight_adjusting_unitdate": False,  # 리밸런싱 시점에 관계없이 매 시점마다 weight 구하는 방법입니다
    "backtest_daily_out": False,  # 월별로 구성된 포트폴리오를 일별로 확장하여 백테스트를 할 것인지 여부
    "backtest_daily_out_lag": [0, 1],  #
    ########### 포트폴리오에서로 구할때는 위에 것들은 따로 설정해줄 필요가 없습니다 ############
    "from_which": "infer",  # infer: infer 데이터로부터 strategy가 진행됩니다. portfolio : universe.csv와 weight.csv가 존재할 경우입니다
    "factor": True,  # factor 와 관련된 백테스트를 할지 여부
    "save_factor": True,  # factor와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    "market_percentile": 0.35,  # 시가총액 상위 몇 %의 주식을 볼지
    "gics": True,  # gics 와 관련된 백테스트를 할지 여부
    "save_gics": True,  # gics와 관련된 정보를 csv 파일로 저장할지 여부 - 그림을 그리기 위해서는 파일로 저장되어 있어야 함
    "gics_level": ["sector"],  # gics 레벨 결정
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