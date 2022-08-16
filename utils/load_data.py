from qraft_data.data import QraftData
from qraft_data.util import get_kirin_api
from pathlib import Path


class LoadData:
    def __init__(self, path, universe):
        """
        initialize universe candidates.
        """
        self.universe = universe
        self._folder_exists_check(path)

    def _folder_exists_check(self, path):
        self.data_path = Path(path)
        self.data_path.mkdir(parents=True, exist_ok=True)

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