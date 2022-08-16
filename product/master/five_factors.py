from pathlib import Path
import pandas as pd
from qraft_data.data import QraftData
from qraft_data.util import get_kirin_api


class LoadFiveFactor:
    names = ("five_factors_momentum", "five_factors_value", "five_factors_size", "five_factors_quality", "five_factors_low_vol")

    def __init__(self, path, universe):
        self.universe = universe
        self._folder_exists_check(path)

    def _folder_exists_check(self, path):
        self.data_path = Path(path)
        self.data_path.mkdir(parents=True, exist_ok=True)

    def call_if_not_loaded(self, name: str) -> QraftData:
        assert name in self.names

        while True:
            try:
                qdata = QraftData.load(name, self.data_path)
                print(f"load_data: {name}")
                return qdata

            except FileNotFoundError:
                print(f"File Not Found: {name}")
                try:
                    api = get_kirin_api(self.universe)
                    if name == "five_factors_momentum":
                        data = self.get_momentum(api)
                    
                    elif name == "five_factors_value":
                        data = self.get_value(api)
                    
                    elif name == "five_factors_size":
                        data = self.get_size(api)
                    
                    elif name == "five_factors_quality":
                        data = self.get_quality(api)
                    
                    elif name == "five_factors_low_vol":
                        data = self.get_low_vol(api)
                    
                    else:
                        raise ValueError

                    qdata = QraftData(name, data)
                    qdata.save(self.data_path.as_posix())
                    print(f"save data: {name}")

                except Exception as e:
                    print(e, name)

            except Exception as e:
                raise e

    def get_momentum(self, api):
        mom13_1 = api.compustat.get_monthly_price_return(12, 0).shift(1)
        mom7_1 = api.compustat.get_monthly_price_return(6, 0).shift(1)
        vol_36 = api.compustat.get_monthly_volatility(36)
        
        # vol adjusted momentum 계산
        mom13_1 = mom13_1 / vol_36
        mom7_1 = mom7_1 / vol_36

        # winsorizing & Z score
        mom13_1 = self.to_z(self.winsorize(mom13_1))
        mom7_1 = self.to_z(self.winsorize(mom7_1))
        mom_score = 0.5*mom13_1 + 0.5*mom7_1

        return mom_score

    def get_value(self, api):
        book_to_market_value = api.high_level.equity.get_book_to_market()
        earnings_to_market_value = api.high_level.equity.get_earnings_to_market()
        dividend_yield = api.high_level.equity.get_dividend_yield()

        book_to_market_value = self.to_z(self.winsorize(book_to_market_value))
        earnings_to_market_value = self.to_z(self.winsorize(earnings_to_market_value))
        dividend_yield = self.to_z(self.winsorize(dividend_yield))

        value_score = 1/3 * book_to_market_value + 1/3 * earnings_to_market_value + 1/3 * dividend_yield
        return value_score

    def get_size(self, api):
        size = -1 * api.compustat.get_monthly_market_value()
        size_score = self.to_z(self.winsorize(size))

        return size_score

    def get_quality(self, api):
        roe = api.high_level.equity.get_roe()
        neg_leverage = -1 * api.high_level.equity.get_leverage()
        neg_earnings_variability = -1 * api.compustat.get_fundamental_data("epsfi").pct_change(12).rolling(60).std()

        roe = self.to_z(self.winsorize(roe))
        neg_leverage = self.to_z(self.winsorize(neg_leverage))
        neg_earnings_variability = self.to_z(self.winsorize(neg_earnings_variability))

        quality_score = 1/3*(roe + neg_leverage + neg_earnings_variability)

        return quality_score

    def get_low_vol(self, api):
        volatility = -1 * api.compustat.get_monthly_volatility(12)
        vol_score = self.to_z(self.winsorize(volatility))

        return vol_score

    @staticmethod
    def to_z(df: pd.DataFrame):
        return df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)

    @staticmethod
    def winsorize(df: pd.DataFrame):
        return df.clip(df.quantile(0.05, axis=1), df.quantile(0.95, axis=1), axis=0)
