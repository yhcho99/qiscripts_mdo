import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from qraft_data.data import QraftData

test_harvest_config = {
        "emissions_fp" : "./emissions_intensity_per_sales.xlsx",
        }

class Harvest:
    def __init__(self, harvest_config, meta):
        self._harvest_config = deepcopy(harvest_config)
        self._emissions_df = self._read_emissions_file(harvest_config)
        self._tickers = set(self._emissions_df.columns)
        #self._api = api
        self._gvkey_iid_to_tic_df = meta[["gvkey_iid", "tic"]].drop_duplicates()\
                .set_index("gvkey_iid").squeeze()
        self._meta = meta.set_index("gvkey_iid")

    def _read_emissions_file(self, harvest_config):
        df = pd.read_excel(Path(harvest_config["emissions_fp"]))
        df = df[df.columns[:3]]
        df = df.rename(
                columns={
                    "Data Date": "date",
                    "GHG/CO2 Emissions Intensity per Sales": "emissions_per_sale",
                    "Ticker": "ticker"
                    })
        df.ticker = df.ticker.str.split().str[0]
        df['date'] = pd.to_datetime(df['date'])
        df = df.pivot_table(index='date', columns="ticker", values="emissions_per_sale")
        return df

    def _gvkey_iid_to_ticker(self, gvkey_iid):
        ticker = self._gvkey_iid_to_tic_df.loc[gvkey_iid]
        if isinstance(ticker, pd.Series):
            # if there are more than two tickers
            ticker_filter = ticker.apply(lambda x: x in self._tickers)
            if ticker_filter.sum() > 1:
                df = self._meta.loc[gvkey_iid]
                tickers = ticker[ticker_filter].tolist()
                df = df.loc[df['tic'].isin(tickers)]
                return df.iloc[df['effdate'].argmax()]['tic']
            if ticker_filter.sum() == 0:
                return np.nan
            return ticker[ticker_filter].item()

        return ticker if ticker in self._tickers else np.nan

    def get_china_filter(self, date_list, gvkey_iid_list):
        CHINA_SYM = "CHN"
        sr = (self._meta["fic"] != CHINA_SYM) & (self._meta["loc"] != CHINA_SYM)
        sr = sr.groupby(sr.index).all()
        sr = sr.reindex(gvkey_iid_list)

        df = pd.DataFrame(index=date_list, columns=gvkey_iid_list, data=True)
        df[sr.index[sr==False]] = False
        df = QraftData("filter_china", df)
        return df

    def get_emissions_filter(self, date_list, gvkey_iid_list, filter_prev=None, exclusion_pct=0.2):
        ticker_list = gvkey_iid_list.map(self._gvkey_iid_to_ticker)
        df = pd.DataFrame(index=date_list, columns=gvkey_iid_list)
        data = self._emissions_df.reindex(date_list)
        idx_ticker_exists = np.nonzero(ticker_list.notna())[0]
        df.iloc[:, idx_ticker_exists] = data.loc[:, ticker_list.dropna()]
        df = df.ffill()

        if filter_prev is not None:
            df = df.where(filter_prev.data)

        df = df.rank(axis=1, ascending=False, pct=True)
        df = QraftData("filter_emission", df)
        return df >= exclusion_pct


