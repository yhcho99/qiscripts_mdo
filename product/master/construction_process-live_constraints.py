# TODO: Basket1 분리 잘 해야 함
"""Check Constraints and adjust weight if needed."""
import sys; sys.path.append(".")
from pathlib import Path
import operator
from pprint import pprint
import numpy as np
import scipy.optimize as sco
import pandas as pd
from kirin import Kirin
import product.master.master_utils as master_utils
from product.master.baskets import BASKET1, BASKET2, MAPPING, REVERSE_MAPPING


BASE_PATH = Path("/home/sronly/sr-storage/master-220228")
START_DATE = "2020-12-15"
FIRST_INITIATION = True
BASKET1_AUM = 30_000_000
BASKET2_AUM = 70_000_000
TRADING_RATIO_THRESHOLD = 0.15
HOLDING_RATIO_THRESHOLD = 0.50
ADV_N = 20

CONSTRAINT_SOURCE = "file"  # "file" or "calculation"
UNIVERSE_FN = "Softbank_Universe-20220225.csv"
RESTRICTED_LIST_FN = "Softbank_Restricted_List-20220225.csv"
ADV_FN = "Softbank_ADVs-20220225.csv"


def reallocate_trade_restricted(target_df, previous_df, trade_restricted_list):
    target_sum = target_df.sum()
    previous_restricted_sum = previous_df.loc[trade_restricted_list].sum()

    adj_factor = (target_sum - previous_restricted_sum) / target_sum
    target_df = adj_factor * reallocate_non_investable(target_df, trade_restricted_list)
    target_df.loc[trade_restricted_list] = previous_df.loc[trade_restricted_list]

    return target_df


def reallocate_non_investable(df, non_investable_index):
    value_sum = df.loc[non_investable_index].sum()
    df.loc[non_investable_index] = 0.0
    df_w = df / df.sum()

    return df + value_sum * df_w


def aggregated_sum(df1: pd.DataFrame, df2: pd.DataFrame):
    df1 = df1.dropna()
    df2 = df2.dropna()
    union_index = df1.index.union(df2.index)
    return df1.reindex(union_index).fillna(0.0) + df2.reindex(union_index).fillna(0.0)


# def calculate_daily_weight(api, fp):
#     basket_weight = pd.read_csv(
#         fp, index_col=0, parse_dates=True
#     ).loc[START_DATE:].dropna(how="all", axis=1)
#     basket_weight = basket_weight.rename(MAPPING, axis=1)

#     new_index = []
#     for date in basket_weight.index:
#         latest_operating_date = master_utils.change_date_to_the_latest_operating_date(date)
#         if date.strftime('%Y-%m-%d') == latest_operating_date.strftime('%Y-%m-%d'):
#             new_index.append(latest_operating_date)
#         else:
#             new_index.append(master_utils.change_date_to_the_lagged_operating_date(date, 1))
#     basket_weight.index = new_index
#     print(basket_weight)

#     gvkey_iid_list = [e.split("_") for e in basket_weight.columns]
#     basket_dividend_adjusted_price = [
#         master_utils.get_daily_price(api, gvkey, iid, adjust_dividend=True, start=START_DATE, end=None)
#         for gvkey, iid in gvkey_iid_list
#     ]
#     basket_dividend_adjusted_price = pd.concat(basket_dividend_adjusted_price, axis=1).ffill()
#     basket_dividend_adjusted_price.columns = basket_weight.columns
#     basket_dividend_adjusted_return = basket_dividend_adjusted_price.pct_change(fill_method=None)

#     daily_basket_weight, _, _ = master_utils.get_daily_portfolio_weight_and_return_from_month_end_weight(
#         basket_weight, basket_dividend_adjusted_return, 0, return_turnover=True
#     )

#     return daily_basket_weight


if __name__ == "__main__":
    assert CONSTRAINT_SOURCE in ("file", "calculation")

    api = Kirin()
    meta = api.compustat.set_investment_universe(
        exchange=["NYSE", "NASDAQ", "OTC"],
        security_type=["COMMON", "ADR"],
        class_A_only=False, primary_issue=False
    )

    # 1. Create Master Notional
    daily_basket1_weight = pd.read_csv(BASE_PATH / "previous_basket1.csv", index_col=0, parse_dates=True).iloc[-1]
    daily_basket2_weight = pd.read_csv(BASE_PATH / "previous_basket2.csv", index_col=0, parse_dates=True).iloc[-1]
    daily_basket1_weight.index = [MAPPING[e] for e in daily_basket1_weight.index]
    daily_basket2_weight.index = [MAPPING[e] for e in daily_basket2_weight.index]
    
    target_basket1_weight = pd.read_csv(BASE_PATH / "strategy" / "combined.csv", index_col=0, parse_dates=True).iloc[-1]
    target_basket2_weight = pd.read_csv(BASE_PATH / "intermediate" / "BASKET2" / "strategy" / "combined.csv", index_col=0, parse_dates=True).iloc[-1]
    
#     daily_basket1_weight = calculate_daily_weight(api, BASE_PATH / "Rebalancing History Basket1.csv")
#     daily_basket2_weight = calculate_daily_weight(api, BASE_PATH / "Rebalancing History Basket2.csv")

    previous_basket1_notional = BASKET1_AUM * daily_basket1_weight
    target_basket1_notional = BASKET1_AUM * target_basket1_weight

    previous_basket2_notional = BASKET2_AUM * daily_basket2_weight
    target_basket2_notional = BASKET2_AUM * target_basket2_weight

    previous_master_notional = aggregated_sum(previous_basket1_notional, previous_basket2_notional)
    target_master_notional = aggregated_sum(target_basket1_notional, target_basket2_notional)

    if FIRST_INITIATION:
        previous_master_notional.loc[:] = 0.0

    # 2. Reorder columns
    columns = previous_master_notional.index.union(target_master_notional.index)
    basket1_locations = [columns.get_loc(e) for e in target_basket1_notional.index]
    target_columns = columns
    previous_master_notional = previous_master_notional.reindex(columns)
    target_master_notional = target_master_notional.reindex(columns)

    # 3. Transform gvkey iid to BBG Ticker
    bbg_tickers = columns.map(lambda e: master_utils.get_latest_bbg_ticker(api, *e.split("_")))
    target_bbg_columns = target_columns.map(lambda e: master_utils.get_latest_bbg_ticker(api, *e.split("_")))
    previous_master_notional.columns = bbg_tickers

    non_included, restricted = pd.Index([]), pd.Index([])
    if CONSTRAINT_SOURCE == "file":

        # 4. Check Universe List
        master_universe_list = pd.read_csv(BASE_PATH / UNIVERSE_FN).iloc[:, 0].to_list()

        if not target_bbg_columns.isin(master_universe_list).all():

            non_included_filter = ~target_bbg_columns.isin(master_universe_list)
            non_included = target_columns[non_included_filter]
            non_included_tickers = target_bbg_columns[non_included_filter]

            print("Some stocks are Non-Included in universe")
            pprint(pd.Series(non_included_tickers, non_included))
            pd.Series(non_included_tickers, non_included).to_csv(BASE_PATH / "universe_non-included.csv")
            target_master_notional = reallocate_non_investable(target_master_notional, non_included)

        else:
            print("All stocks are Included in universe")

        # 5. Check Restricted List
        restricted_list = pd.read_csv(BASE_PATH / RESTRICTED_LIST_FN).iloc[:, 0].to_list()

        if target_bbg_columns.isin(restricted_list).any():

            restricted_filter = target_bbg_columns[target_bbg_columns.isin(restricted_list)]
            restricted = target_columns[restricted_filter]
            restricted_tickers = target_bbg_columns[restricted_filter]

            print("Some stocks are restricted.")
            pprint(pd.Series(restricted_tickers, restricted))
            pd.Series(restricted_tickers, restricted).to_csv(BASE_PATH / "restricted_list-included.csv")
            target_master_notional = reallocate_trade_restricted(
                target_master_notional, previous_master_notional, restricted
            )
        else:
            print("All stocks are non-restricted.")

        # 6. Get ADV
        adv_series = pd.read_csv(BASE_PATH / ADV_FN, index_col=0).iloc[:, 0].astype(float)
        file_included = bbg_tickers.intersection(adv_series.index)

        master_adv = {}
        for gvkey_iid, bbg in zip(columns, bbg_tickers):
            if bbg in file_included:
                master_adv[gvkey_iid] = adv_series.loc[bbg]
            else:
                print(f"ADV non-included ({gvkey_iid}, {bbg})")
                gvkey, iid = gvkey_iid.split("_")
                master_adv[gvkey_iid] = master_utils.get_latest_adv(api, gvkey, iid, ADV_N)

    else:
        # 6. Get ADV
        print("Calculation constraint source not check universe inclusion.")
        master_adv = {}
        for gvkey_iid in columns:
            gvkey, iid = gvkey_iid.split("_")
            master_adv[gvkey_iid] = master_utils.get_latest_adv(api, gvkey, iid, ADV_N)

    # 7. Optimization
    master_adv = pd.Series(master_adv)

    master_non_optimizable_index = non_included.union(restricted)
    master_non_optimizables = target_master_notional.loc[master_non_optimizable_index]

    master_optimizable_index = columns.difference(master_non_optimizable_index)
    master_optimizable_target_values = target_master_notional.loc[master_optimizable_index].fillna(0.0).values.copy()
    master_optimizable_previous_values = previous_master_notional.loc[master_optimizable_index].fillna(0.0).values.copy()
    master_optimizable_adv = master_adv.loc[master_optimizable_index].values.copy()

    if (
            master_utils.is_satisfying_liquidity_constraint(
                master_optimizable_target_values,
                master_optimizable_adv,
                HOLDING_RATIO_THRESHOLD,
                in_optimization=False
            ) and master_utils.is_satisfying_liquidity_constraint(
                master_optimizable_target_values - master_optimizable_previous_values,
                master_optimizable_adv,
                TRADING_RATIO_THRESHOLD,
                in_optimization=False
            )
    ):
        print("Skip optimization")
        optimized_values = master_optimizable_target_values

    else:
        print("Do optimization")

        res = sco.minimize(
            lambda x: np.mean(np.square(x - master_optimizable_target_values)),
            x0=master_optimizable_target_values.copy(),
            bounds=[(0.0, BASKET1_AUM + BASKET2_AUM) for _ in range(len(master_optimizable_target_values))],
            constraints=[
                {"type": "eq", "fun": lambda e: e.sum() - BASKET1_AUM + BASKET2_AUM},
                {"type": "ineq", "fun": lambda e: master_utils.is_satisfying_liquidity_constraint(
                    e, master_optimizable_adv, HOLDING_RATIO_THRESHOLD, in_optimization=True)
                 },
                {"type": "ineq", "fun": lambda e: master_utils.is_satisfying_liquidity_constraint(
                    e - master_optimizable_previous_values, master_optimizable_adv, TRADING_RATIO_THRESHOLD, in_optimization=True)
                 },
                {"type": "ineq", "fun": lambda e: e[basket1_locations].sum() - BASKET1_AUM}
            ],
            tol=100
        )
        optimized_values = res.x
        if not res.success:
            import warnings
            warnings.warn("Optimization Failed.")

            if not master_utils.is_satisfying_liquidity_constraint(
                    optimized_values, master_optimizable_adv, HOLDING_RATIO_THRESHOLD, in_optimization=False
            ):
                print("Holding Liquidation Failed")

            if not master_utils.is_satisfying_liquidity_constraint(
                    optimized_values - master_optimizable_previous_values, master_optimizable_adv, TRADING_RATIO_THRESHOLD, in_optimization=False
            ):
                print("Trading Liquidation Failed")

            if not (optimized_values >= 0).all():
                ValueError("Some negative")

            if optimized_values[basket1_locations].sum() < BASKET1_AUM:
                ValueError("Basket1 Allocation Error")

    # 8. Reorder
    optimized_master_notional = pd.concat([pd.Series(optimized_values, master_optimizable_index), master_non_optimizables])
    optimized_master_notional = optimized_master_notional.loc[columns]

    # 9. Re-split
    # TODO: Split 시 원래 Basket 1/2 비중과 최대한 가깝게 다시 분배하는 최적화 과정 필요
    
    basket1_shared = optimized_master_notional.iloc[basket1_locations].copy()
    optimized_basket1_notional = BASKET1_AUM * basket1_shared / basket1_shared.sum()
    optimized_basket1_weight = optimized_basket1_notional / optimized_basket1_notional.sum()
    print("Basket1 Notional", optimized_basket1_notional.sum())

    optimized_basket1_notional.to_csv(BASE_PATH / "latest_basket1(notional).csv")
    optimized_basket1_weight.to_csv(BASE_PATH / "latest_basket1(weight).csv")

    optimized_basket2_notional = optimized_master_notional.copy()
    optimized_basket2_notional.iloc[basket1_locations] -= optimized_basket1_notional
    optimized_basket2_weight = optimized_basket2_notional / optimized_basket2_notional.sum()
    print("Basket2 Notional", optimized_basket2_notional.sum())

    optimized_basket2_notional.to_csv(BASE_PATH / "latest_basket2(notional).csv")
    optimized_basket2_weight.to_csv(BASE_PATH / "latest_basket2(weight).csv")
