import warnings
from typing import List, Tuple, Union, Optional
from pathlib import Path
import multiprocessing
import os
import os.path
from collections import defaultdict
import pandas as pd
import datetime
from pandas.tseries.offsets import MonthEnd

from qraft_data.data import QraftData
from qraft_data.util import MultiProcess, get_compustat_api
from qraft_data.universe import get_universe
from qraft_data.loader import FactorFactoryDataDownLoader


class FFManager:
    def __init__(
        self,
        date_from: str,
        date_to: str,
        rebalancing_terms: str,
        universe_names: List[str],
        description: str,
        metrics: List[Tuple[str, int]],
        num_factors: int,
        kirin_cache_dir: str,
        factor_save_path: str,
        pretend_monthend: Union[bool, str],
        cpu_count: int,
        reset: bool = False,
    ):
        self.date_from = (pd.Timestamp(date_from) - MonthEnd(1)).strftime("%Y-%m-%d")
        self.date_to = (pd.Timestamp(date_to) - MonthEnd(1)).strftime("%Y-%m-%d")
        self.rebalancing_terms = rebalancing_terms
        period_list = (
            pd.date_range(
                start=self.date_from,
                end=self.date_to,
                freq=self.rebalancing_terms,
            )
            .to_pydatetime()
            .tolist()
        )
        self.period_list = [d.strftime("%Y-%m-%d") for d in period_list]


        self.universe_names = universe_names
        self.default_arguments = {
            "description": description,
            "metrics": metrics,
            "num_factors": num_factors,
            "kirin_cache_dir": kirin_cache_dir,
            "factor_save_path": factor_save_path,
        }

        self.pretend_monthend = pretend_monthend
        self.downloader = FactorFactoryDataDownLoader(num_parallel=6)
        self.cpu_count = cpu_count
        self.run_proc = MultiProcess()
        self.run_proc.cpu_count(
            max_count=self.cpu_count * len(self.universe_names)
        )
        self._can_be_start = None
        self.__columns = None
        self.reset = reset
        if reset:
            os.rmdir(self.default_arguments["kirin_cache_dir"])
            os.rmdir(self.default_arguments["factor_save_path"])
        Path(self.default_arguments["kirin_cache_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.default_arguments["factor_save_path"]).mkdir(parents=True, exist_ok=True)

    # 1. Mainly used
    def prepare_experiment(self):
        assert self._can_be_start is None
        # runs for ~20 minutes
        if len(os.listdir(self.default_arguments["kirin_cache_dir"])) < 150:
            self.__download_kirin_datas(self.universe_names[0], num_parallel=30)
        #if len(os.listdir(self.default_arguments["factor_save_path"])) == 0:
        self.__download_factor_datas(end_dates=self.period_list, num_parallel=6)
        self._can_be_start = "experiment"



    # 2. Mainly used
    def prepare_inference(self):
        assert self._can_be_start is None
        if len(os.listdir(self.default_arguments["kirin_cache_dir"])) < 150:
            self.__download_kirin_datas(self.universe_names[0], num_parallel=30)
        self.__download_factor_datas(end_dates=self.period_list[:self.cpu_count], num_parallel=6)
        self._can_be_start = "inference"

    # 3. Mainly used
    def prepare_is_checked(self):
        assert self._can_be_start is None
        assert len(os.listdir(self.default_arguments["kirin_cache_dir"])) > 0 
        assert len(os.listdir(self.default_arguments["factor_save_path"])) > 0
        self._can_be_start = "checked"

    # 4. Mainly used
    def datas_on_date(self, date):
        assert (
            self._can_be_start is not None
        ), "Not prepared to start, pleas call self.prepare_experiment or self.prepare_inference first."
        assert not isinstance(date, str)
        end_date = date - MonthEnd(1)
        next_date = end_date + MonthEnd(self.cpu_count)
        end_date = end_date.strftime("%Y-%m-%d")
        next_date = next_date.strftime("%Y-%m-%d")
        if next_date <= self.date_to:
            self.__pipelined_preparation(next_date)

        kargs = self.default_arguments.copy()
        kargs.pop("kirin_cache_dir")
        ffs = self.downloader.load_factors(
            universe_names=self.universe_names,
            end_date=end_date,
            return_as_list=True,
            **kargs,
        )
        return ffs
        # preprocessed_ffs = []
        # for f in ffs:
        #     ff = 2 * f.rank(pct=True) - 1
        #     preprocessed_ffs.append(ff)
        # return preprocessed_ffs

    @property
    def columns(self):
        if self.__columns is None:
            self.__make_columns()
        return self.__columns

    def __make_columns(self):
        assert (
            self._can_be_start is not None
        ), "Not prepared to start, pleas call self.prepare_experiment or self.prepare_inference first."

        factor_save_path = Path(self.default_arguments["factor_save_path"])
        if self._can_be_start == "inference":
            period_list = self.period_list[:self.cpu_count]
        else: 
            period_list = self.period_list

        list_of_oldest_filepath_universe = []
        for univ_name in self.universe_names:
            universe = get_universe(univ_name)
            universe_id = self.downloader.univ_conn.get_universe_id(universe)
            id_list, _ = self.downloader._factor_factory_data_list(
                universe_id=universe_id,
                description=self.default_arguments["description"],
                metrics=self.default_arguments["metrics"],
                num_factors=self.default_arguments["num_factors"],
                end_dates=period_list,
            )
            for _id in id_list:
                filepath = factor_save_path / univ_name / f"{_id}.pkl"
                list_of_oldest_filepath_universe.append(filepath)

        oldest_filepath = self.__get_oldest_filepath(
            list_of_oldest_filepath_universe
        )
        oldest_data = QraftData.load(
            oldest_filepath.stem, oldest_filepath.parent
        )

        self.__columns = oldest_data.data.columns

    @staticmethod
    def __get_oldest_filepath(filepath_list) -> Path:
        filepath_list.sort(key=lambda e: os.path.getmtime(e))

        return filepath_list[0]

    def __download_kirin_datas(self, univ_name: str, num_parallel: int = 30) -> None:
        self.downloader.num_parallel = num_parallel
        kirin_cache_dir = self.default_arguments.copy()["kirin_cache_dir"]
        self.downloader.download_kirin_datas(
            univ_name=univ_name,
            kirin_cache_dir=kirin_cache_dir,
            pretend_monthend=self.pretend_monthend,
        )

    def __download_factor_datas(
        self, end_dates: Optional[list], num_parallel: int = 6
    ) -> None:
        self.downloader.num_parallel = num_parallel
        self.downloader.download_factor_datas(
            univ_names=self.universe_names,
            end_dates=end_dates,
            pretend_monthend=self.pretend_monthend,
            reset=self.reset,
            **self.default_arguments.copy(),
        )

    def __pipelined_preparation(self, end_date: str) -> None:
        assert isinstance(end_date, str)

        for univ_name in self.universe_names:
            args = (univ_name, end_date)
            proc = multiprocessing.Process(
                target=self.__prepare_datas, args=args
            )
            self.run_proc.run_process(proc)
        print(f"[{end_date}] ff datas are prepared")

    def __prepare_datas(self, univ_name: str, end_date: str) -> None:
        assert isinstance(end_date, str)
        if self.downloader.is_already_downloaded(
            univ_name=univ_name,
            end_dates=[end_date],
            **self.default_arguments.copy(),
        ):
            pass
        else:
            print(f"[Not Yet Downloaded] {end_date}, {univ_name}")
            self.downloader.download_factor_data(
                univ_name=univ_name,
                end_dates=[end_date],
                pretend_monthend=self.pretend_monthend,
                reset=self.reset,
                **self.default_arguments.copy(),
            )
