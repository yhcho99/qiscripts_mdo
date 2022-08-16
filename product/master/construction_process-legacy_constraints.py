"""Check Constraints and adjust weight if needed."""
import sys; sys.path.append(".")
from pathlib import Path
import numpy as np
import scipy.optimize as sco
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from paths import DATA_DIR


IDENTIFIER = "master-220201(Preweight-220125)"
AUM = 100_000_000
TRADING_RATIO_THRESHOLD = 0.15
HOLDING_RATIO_THRESHOLD = 0.50
ADV_N = 20


if __name__ == "__main__":
    exp_path = Path(DATA_DIR) / IDENTIFIER

    # 1. Read Target Portfolio
    print("Read Target Portfolio")
    latest_basket1_path = exp_path / "strategy" / "combined.csv"
    latest_basket2_path = exp_path / "intermediate" / "BASKET2" / "strategy" / "combined.csv"

    latest_basket1 = pd.read_csv(latest_basket1_path, index_col=0, parse_dates=True).iloc[-1].dropna()
    latest_basket2 = pd.read_csv(latest_basket2_path, index_col=0, parse_dates=True).iloc[-1].dropna()

    previous_basket1_path = exp_path / "previous_basket1.csv"
    previous_basket2_path = exp_path / "previous_basket2.csv"

    # 2. Read Previous Portfolio
    print("Read Previous Portfolio")
    previous_basket1 = pd.read_csv(previous_basket1_path, index_col=0).iloc[:, 0]
    previous_basket2 = pd.read_csv(previous_basket2_path, index_col=0).iloc[:, 0]

    print("Transform BBG Ticker to GVKEY IID")
    api = Kirin()
    previous_basket1.index = previous_basket1.index.map(lambda e: master_utils.get_gvkey_iid_from_latest_bbg_ticker(api, e))
    previous_basket2.index = previous_basket2.index.map(lambda e: master_utils.get_gvkey_iid_from_latest_bbg_ticker(api, e))

    # 3. Check if there are FAIL cases during transforming BBG Ticker to gvkey iid
    previous_basket1_fail_cases = previous_basket1.reindex([e for e in previous_basket1.index if e.startswith("FAIL-")])
    previous_basket2_fail_cases = previous_basket2.reindex([e for e in previous_basket2.index if e.startswith("FAIL-")])

    if len(previous_basket1_fail_cases):
        print("Fail some cases for transforming BBG Ticker to gvkey iid in previous BASKET1")
        previous_basket1_fail_cases.to_csv(exp_path / "fail-cases_previous_basket1.csv")
        previous_basket1 = previous_basket1.loc[previous_basket1.index.difference(previous_basket1_fail_cases.index)]

    if len(previous_basket2_fail_cases):
        print("Fail some cases for transforming BBG Ticker to gvkey iid in previous BASKET2")
        previous_basket2_fail_cases.to_csv(exp_path / "fail-cases_previous_basket2.csv")
        previous_basket2 = previous_basket2.loc[previous_basket2.index.difference(previous_basket2_fail_cases.index)]

    # 4. Check BASKET1 target portfolios satisfy liquidity constraints.
    print("Check BASKET1")
    union_index_for_basket1 = latest_basket1.index.union(previous_basket1.index)
    latest_basket1_holdings = AUM * latest_basket1.reindex(union_index_for_basket1).fillna(0.0)
    previous_basket1_holdings = AUM * previous_basket1.reindex(union_index_for_basket1).fillna(0.0)
    basket1_adv = np.array(
        [master_utils.get_latest_adv(api, e.split("_")[0], e.split("_")[1], ADV_N) for e in union_index_for_basket1],
        dtype=np.float64
    )
    basket1_trading_required = (latest_basket1_holdings - previous_basket1_holdings).values.copy()
    basket1_holdings_required = latest_basket1_holdings.values.copy()

    if (
            master_utils.is_satisfying_liquidity_constraint(basket1_trading_required, basket1_adv, TRADING_RATIO_THRESHOLD, in_optimization=False)
            and master_utils.is_satisfying_liquidity_constraint(basket1_holdings_required, basket1_adv, HOLDING_RATIO_THRESHOLD, in_optimization=False)
    ):
        print("TARGET BASKET1 satisfies liquidity constraints.")
        (latest_basket1 / latest_basket1.sum()).to_csv(exp_path / "latest_basket1.csv")
    else:
        # 4-1. Correct BASKET1 if it does not satisfy liquidity.
        print("TARGET BASKET1 needs correction.")
        basket1_ideal_holdings = latest_basket1_holdings.values.copy()
        basket1_as_is_holdings = previous_basket1_holdings.values.copy()
        res = sco.minimize(
            lambda x: np.mean(np.square(x - basket1_ideal_holdings)),
            x0=basket1_ideal_holdings.copy(),
            bounds=[(0.0, AUM) for _ in range(len(basket1_ideal_holdings))],
            constraints=[
                {"type": "eq", "fun": lambda e: e.sum() - AUM},
                {"type": "ineq", "fun": lambda e: master_utils.is_satisfying_liquidity_constraint(
                    e, basket1_adv, HOLDING_RATIO_THRESHOLD, in_optimization=True)
                 },
                {"type": "ineq", "fun": lambda e: master_utils.is_satisfying_liquidity_constraint(
                    e - basket1_as_is_holdings, basket1_adv, HOLDING_RATIO_THRESHOLD, in_optimization=True)
                 },
            ]
        )
        if not res.success:
            raise ValueError("Optimization Failed in Basket1.")
        else:
            print("TARGET BASKET1 is successfully corrected")
            latest_basket1 = pd.Series(res.x, union_index_for_basket1, dtype=np.float64)
            (latest_basket1 / latest_basket1.sum()).to_csv(exp_path / "latest_basket1.csv")

    # 5. Check BASKET2 target portfolios satisfy liquidity constraints.
    print("Check BASKET2")
    union_index_for_basket2 = latest_basket2.index.union(previous_basket2.index)
    latest_basket2_holdings = AUM * latest_basket2.reindex(union_index_for_basket2).fillna(0.0)
    previous_basket2_holdings = AUM * previous_basket2.reindex(union_index_for_basket2).fillna(0.0)
    basket2_adv = np.array(
        [master_utils.get_latest_adv(api, e.split("_")[0], e.split("_")[1], ADV_N) for e in union_index_for_basket2],
        dtype=np.float64
    )
    basket2_trading_required = (latest_basket2_holdings - previous_basket2_holdings).values.copy()
    basket2_holdings_required = latest_basket2_holdings.values.copy()

    if (
            master_utils.is_satisfying_liquidity_constraint(basket2_trading_required, basket2_adv, TRADING_RATIO_THRESHOLD, in_optimization=False)
            and master_utils.is_satisfying_liquidity_constraint(basket2_holdings_required, basket2_adv, HOLDING_RATIO_THRESHOLD, in_optimization=False)
    ):
        print("TARGET BASKET2 satisfies liquidity constraints.")
        (latest_basket2 / latest_basket2.sum()).to_csv(exp_path / "latest_basket2.csv")
    else:
        # 5-1. Correct BASKET2 if it does not satisfy liquidity.
        print("TARGET BASKET2 needs correction.")
        basket2_ideal_holdings = latest_basket2_holdings.values.copy()
        basket2_as_is_holdings = previous_basket2_holdings.values.copy()
        res = sco.minimize(
            lambda x: np.mean(np.square(x - basket2_ideal_holdings)),
            x0=basket2_ideal_holdings.copy(),
            bounds=[(0.0, AUM) for _ in range(len(basket2_ideal_holdings))],
            constraints=[
                {"type": "eq", "fun": lambda e: e.sum() - AUM},
                {"type": "ineq", "fun": lambda e: master_utils.is_satisfying_liquidity_constraint(
                    e, basket2_adv, HOLDING_RATIO_THRESHOLD, in_optimization=True)
                 },
                {"type": "ineq", "fun": lambda e: master_utils.is_satisfying_liquidity_constraint(
                    e - basket2_as_is_holdings, basket2_adv, HOLDING_RATIO_THRESHOLD, in_optimization=True)
                 },
            ]
        )
        if not res.success:
            raise ValueError("Optimization Failed in Basket2.")
        else:
            print("TARGET BASKET2 is successfully corrected")
            latest_basket2 = pd.Series(res.x, union_index_for_basket2, dtype=np.float64)
            (latest_basket2 / latest_basket2.sum()).to_csv(exp_path / "latest_basket2.csv")
