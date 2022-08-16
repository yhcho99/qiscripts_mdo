import pandas as pd
import numpy as np

from qraft_data.loader import VIFDataLoader, SOTADataLoader
from qraft_data.loader import UniverseConn

# universe_info can be universe_id, config, or object.

universe_info = "5fece25cd9a279f21dad0a00"
sota_data_loader = SOTADataLoader()
sota_datas = sota_data_loader.load(universe_info)
for d in sota_datas:
    print(d.name)
    print(d.data)