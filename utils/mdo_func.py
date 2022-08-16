from qraft_data.data import QraftData
from qraft_data.util import get_kirin_api
from pathlib import Path

import pandas as pd
import numpy as np


class Mdo:
    def __init__(self, universe) -> None:

        class MdoFunc:
            def __init__(self) -> None:
                pass

            """
            이하 포멧으로 바꿔줌

            Return         Stock A     Stock B     Stock C
            2020/1/31       1.3         2           4
            2020/2/28...    1           3           5
            2020/3/31...
            2020/4/30...    NaN
            ...

            """
            def pivot_table(self, df, values='factor', columns='security'):
                pivoted = df.pivot_table(values=values, index=df.index, columns=columns, aggfunc='first')
                return pivoted
            
            def inf_to_nan(self, df):
                #df[:] ensures that df doesn't convert to np array
                df[:] = np.where(np.isfinite(df), df, np.nan)
                return df

            def add_security_names(self, date_from='2000-1-1', date_to='2022-8-1', factor_series=None) -> pd.DataFrame:
                factor_series.name = 'factor'
                securities = self.universe.data.loc[date_from:date_to, 'security']
                return pd.concat([securities, factor_series], axis=1)

            def get_monthly_price_return(self, period, date_from='2000-1-1', date_to='2022-8-1', column='DS_CLOSE'):
                price = self.universe.data.loc[date_from - pd.offsets.MonthEnd(period) : date_to, column]
                ret = price / price.shift(period) - 1
                ret = self.pivot_table(self.add_security_names(date_from, date_to, ret))

                return ret

            def _get_monthly_price_return(self, period, date_from='2000-1-1', date_to='2022-8-1', column='DS_CLOSE'):
                price = self.universe.data.loc[date_from - pd.offsets.MonthEnd(period) : date_to, column]
                ret = price / price.shift(period) - 1

                return ret

            def get_price_momentum(self, mom_start, mom_end, date_from='2000-1-1', date_to='2022-8-1', column='DS_CLOSE'):
                price = self.universe.data.loc[date_from - pd.offsets.MonthEnd(mom_start) : date_to, column]
                mom = price.shift(mom_end) / price.shift(mom_start) - 1
                mom = self.pivot_table(self.add_security_names(date_from, date_to, mom))

                return mom

            def get_monthly_market_value(self, date_from='2000-1-1', date_to='2022-8-1', column='CS_MKVALT_A'):
                mv = self.universe.data.loc[date_from:date_to, column]
                mv = self.pivot_table(self.add_security_names(date_from, date_to, mv))

                return mv

            def get_book_to_market(self, date_from='2000-1-1', date_to='2022-8-1', column=['CS_BKVLPS_A', 'DS_CLOSE']):
                df = self.universe.data.loc[date_from:date_to, column]
                btm = df['CS_BKVLPS_A'] / df['DS_CLOSE']
                btm = self.pivot_table(self.add_security_names(date_from, date_to, btm))

                return btm

            def get_ram(self, period, date_from='2000-1-1', date_to='2022-8-1'):
                mom = self._get_monthly_price_return(period, date_from, date_to)
                vol = self._get_monthly_price_return(1, date_from, date_to).rolling(period).std(ddof=0)
                
                ram = self.pivot_table(self.add_security_names(date_from, date_to, mom/vol))

                return ram
            
            def get_monthly_volatility(self, rolling_window=36, date_from='2000-1-1', date_to='2022-8-1'):
                ret = self._get_monthly_price_return(1, date_from, date_to)
                vol = ret.rolling(rolling_window).std(ddof=0).iloc[rolling_window-1:]

                vol = self.pivot_table(self.add_security_names(date_from, date_to, vol))
                
                return vol

            def get_asset_turnover(self, date_from='2000-1-1', date_to='2022-8-1'):
                sale = self.universe.data.loc[date_from:date_to, 'CS_SALE_A']
                ppent = self.universe.data.loc[date_from:date_to, 'CS_PPENT_A']
                asset = self.universe.data.loc[date_from:date_to, 'CS_ACT_A']
                liab = self.universe.data.loc[date_from:date_to, 'CS_LCT_A']
                
                factor = sale / (ppent + asset - liab)
                factor = self.pivot_table(self.add_security_names(date_from, date_to, factor))

                return factor

            def get_gpa(self, date_from='2000-1-1', date_to='2022-8-1'):
                sale = self.universe.data.loc[date_from:date_to, 'CS_SALE_A']
                cogs = self.universe.data.loc[date_from:date_to, 'CS_COGS_A']
                gp = sale - cogs
                at = self.universe.data.loc[date_from:date_to, 'CS_AT_A']
                
                gpa = self.pivot_table(self.add_security_names(date_from, date_to, gp/at))
                return self.inf_to_nan(gpa)

            def get_revenue_surprise(self, date_from='2000-1-1', date_to='2022-8-1'):
                saleq = self.universe.data.loc[date_from:date_to, 'CS_SALEQ_Q']
                cshprq = self.universe.data.loc[date_from:date_to, 'CS_CSHPRQ_Q']
                ajexq = self.universe.data.loc[date_from:date_to, 'CS_AJEXQ_Q']
                
                rps = saleq / (cshprq * ajexq)
                yoy = rps - rps.shift(12)
                rs = yoy / yoy.rolling(48).std()
                rs = self.pivot_table(self.add_security_names(date_from, date_to, rs))
    
                return rs

            def get_cash_to_asset(self, date_from='2000-1-1', date_to='2022-8-1'):
                cheq = self.universe.data.loc[date_from:date_to, "CS_CHQ_Q"]
                atq = self.universe.data.loc[date_from:date_to, "CS_ATQ_Q"]
                
                cta = self.pivot_table(self.add_security_names(date_from, date_to, cheq/atq))
                return self.inf_to_nan(cta)

            def get_operating_leverage(self, date_from='2000-1-1', date_to='2022-8-1'):
                xsga = self.universe.data.loc[date_from:date_to, "CS_XSGA_A"]
                cogs = self.universe.data.loc[date_from:date_to, "CS_COGS_A"]
                at = self.universe.data.loc[date_from:date_to, 'CS_AT_A']
    
                factor = (xsga + cogs) / at
                factor = self.pivot_table(self.add_security_names(date_from, date_to, factor))

                return self.inf_to_nan(factor)

            def get_roe(self, date_from='2000-1-1', date_to='2022-8-1'):
                ni = self.universe.data.loc[date_from:date_to, 'CS_NIQ_Q']
                seq = self.universe.data.loc[date_from:date_to, 'CS_SEQQ_Q'] #annual doesn't exist
                roe = ni/seq
                roe = self.pivot_table(self.add_security_names(date_from, date_to, roe))

                return self.inf_to_nan(roe)

            def get_standardized_unexpected_earnings(self, date_from='2000-1-1', date_to='2022-8-1'):
                eps = self.universe.data.loc[date_from:date_to, 'CS_EPSPXQ_Q']
                ajexq = self.universe.data.loc[date_from:date_to, 'CS_AJEXQ_Q']
                
                adj_eps = eps / ajexq
                yoy = adj_eps - adj_eps.shift(12)
                sue = (yoy - yoy.rolling(9999, min_periods=1).mean()) / yoy.rolling(9999, min_periods=1).std()
                sue = self.pivot_table(self.add_security_names(date_from, date_to, sue))

                return sue

            def get_return_on_net_operating_asset(self, date_from='2000-1-1', date_to='2022-8-1'):
                oiadp = self.universe.data.loc[date_from:date_to, 'CS_OIADP_A']
                
                ppent = self.universe.data.loc[date_from:date_to, 'CS_PPENT_A']
                asset = self.universe.data.loc[date_from:date_to, 'CS_ACT_A']
                liab = self.universe.data.loc[date_from:date_to, 'CS_LCT_A']
                ona = ppent + asset - liab
                
                ret_ona = self.pivot_table(self.add_security_names(date_from, date_to, oiadp/ona))

                return ret_ona

            def get_earnings_to_market(self, date_from='2000-1-1', date_to='2022-8-1'):
                e = self.universe.data.loc[date_from:date_to, 'CS_IB_A']
                mv = self.universe.data.loc[date_from:date_to, 'CS_MKVALT_A']
                
                etm = self.pivot_table(self.add_security_names(date_from, date_to, e/mv))
                return self.inf_to_nan(etm)

            def get_linear_assumed_intangible_asset_to_market_value(self, date_from='2000-1-1', date_to='2022-8-1'):
                xrd = self.universe.data.loc[date_from:date_to, 'CS_XRD_A']
                xsga = self.universe.data.loc[date_from:date_to, 'CS_XSGA_A']
                xad = self.universe.data.loc[date_from:date_to, 'CS_XAD_A']
                mv = self.universe.data.loc[date_from:date_to, 'CS_MKVALT_A']
                
                rc = xrd + 0.8 * xrd.shift(12) + 0.6 * xrd.shift(24) + 0.4 * xrd.shift(36) + 0.2 * xrd.shift(48)
                xsga = xsga + 0.8 * xsga.shift(12) + 0.6 * xsga.shift(24) + 0.4 * xsga.shift(36) + 0.2 * xsga.shift(48)
                xad = xad + 0.8 * xad.shift(12) + 0.6 * xad.shift(24) + 0.4 * xad.shift(36) + 0.2 * xad.shift(48)
                rc = rc[rc >= 0]
                xsga = xsga[xsga >= 0]
                xad = xad[xad >= 0]
                factor = (rc + xsga * 0.8 + xad * 0.5) / mv
                factor = self.pivot_table(self.add_security_names(date_from, date_to, factor))
                
                return factor

            def get_linear_assumed_intangible_asset_to_total_asset(self, date_from='2000-1-1', date_to='2022-8-1'):
                xrd = self.universe.data.loc[date_from:date_to, 'CS_XRD_A']
                xsga = self.universe.data.loc[date_from:date_to, 'CS_XSGA_A']
                at = self.universe.data.loc[date_from:date_to, 'CS_AT_A']
                
                rc = xrd + 0.8 * xrd.shift(12) + 0.6 * xrd.shift(24) + 0.4 * xrd.shift(36) + 0.2 * xrd.shift(48)
                xsga = xsga + 0.8 * xsga.shift(12) + 0.6 * xsga.shift(24) + 0.4 * xsga.shift(36) + 0.2 * xsga.shift(48)
                rc = rc[rc >= 0]
                xsga = xsga[xsga >= 0]
                factor = (rc + xsga * 0.2) / at
                factor = self.pivot_table(self.add_security_names(date_from, date_to, factor))

                return factor
            
            def get_advertising_expense_to_market(self, date_from='2000-1-1', date_to='2022-8-1'):
                xad = self.universe.data.loc[date_from:date_to, 'CS_XAD_A']
                mv = self.universe.data.loc[date_from:date_to, 'CS_MKVALT_A']
                
                factor = xad / mv
                factor = factor[factor > 0]
                factor = self.pivot_table(self.add_security_names(date_from, date_to, factor))

                return self.inf_to_nan(factor)

            def get_rnd_capital_to_asset(self, date_from='2000-1-1', date_to='2022-8-1'):
                xrd = self.universe.data.loc[date_from:date_to, 'CS_XRD_A']
                at = self.universe.data.loc[date_from:date_to, 'CS_AT_A']
                
                rnd_cap = xrd + 0.8 * xrd.shift(12) + 0.6 * xrd.shift(24) + 0.4 * xrd.shift(36) + 0.2 * xrd.shift(48)
                rc = rnd_cap[rnd_cap > 0]

                factor = rc / at
                factor = self.pivot_table(self.add_security_names(date_from, date_to, factor))

                return self.inf_to_nan(factor)

            def get_rnd_to_sale(self, date_from='2000-1-1', date_to='2022-8-1'):
                xrd = self.universe.data.loc[date_from:date_to, 'CS_XRD_A']
                sale = self.universe.data.loc[date_from:date_to, 'CS_SALE_A']
                xrd = xrd[xrd > 0.]
                
                factor = xrd / sale
                factor = self.pivot_table(self.add_security_names(date_from, date_to, factor))

                return self.inf_to_nan(factor)
                
            def get_rnd_to_asset(self, date_from='2000-1-1', date_to='2022-8-1'):
                rnd = self.universe.data.loc[date_from:date_to, 'CS_XRD_A']
                at = self.universe.data.loc[date_from:date_to, 'CS_AT_A']

                rta = self.pivot_table(self.add_security_names(date_from, date_to, rnd/at))
                return self.inf_to_nan(rta)

        self.mdo_func = MdoFunc()
        self.universe = universe
    
    def call_if_not_loaded(self, name: str, method_names: list, args=tuple(), kwargs=None) -> QraftData:
        while True:
            if kwargs is None:
                kwargs = {}
            try:
                qdata = QraftData.load(name, self.data_path)
                print(f"load_data: {name}")
                return qdata

            except FileNotFoundError:
                print(f"FileNotFound: {name}")
                try:
                    if method_names == "trading":
                        api = get_kirin_api(self.universe)
                        data = api.compustat.get_monthly_price_data(adjust_for_split=False, adjust_for_total_return=False) * api.compustat.get_monthly_volume_data()
                    else:
                        obj = get_kirin_api(self.universe)

                        for method_name in method_names: # method_name : ['high_level', 'equity', 'get_monthly_price_return']: 끝에 얻고자 하는 데이터의 주기가 표시되어 있음
                            obj = getattr(obj, method_name)
                        data = obj(*args, **kwargs)
                    qdata = QraftData(name, data)
                    qdata.save(self.data_path.as_posix())
                    print(f"save data: {name}")

                except Exception as e:
                    print(e, name)

            except Exception as e:
                raise e