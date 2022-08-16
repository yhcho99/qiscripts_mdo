import os
import shutil
from pathlib import Path

import pandas as pd

DST_DIR = Path("/home/sronly/sr-storage/zip")
if DST_DIR.is_dir():
    shutil.rmtree(DST_DIR)
DST_DIR.mkdir()


EXP = "ff+wo_alt"
TRY = 1
ACTs = ["0.1500", "0.3000", "0.5000", "0.7000", "0.9900"]

basket1_pr = {}
basket1_tr = {}
basket2_pr = {}
basket2_tr = {}
for act in ACTs:
    st_path = Path("/home/sronly/sr-storage") / f"master_phase2-{EXP}-try{TRY}-ACT_CF{act}"

    b1_path = st_path / "strategy" / "performance.csv"
    b1 = pd.read_csv(b1_path, index_col=0, parse_dates=True)
    basket1_pr[act] = b1["price_return"] - 0.001*b1["turnover_ratio"]
    basket1_tr[act] = b1["total_return"] - 0.001*b1["turnover_ratio"]

    b2_path = st_path / "intermediate" / "BASKET2" / "strategy" / "performance.csv"
    b2 = pd.read_csv(b2_path, index_col=0, parse_dates=True)
    basket2_pr[act] = b2["price_return"] - 0.001*b2["turnover_ratio"]
    basket2_tr[act] = b2["total_return"] - 0.001*b2["turnover_ratio"]

b1_base = pd.read_csv("/home/sronly/sr-storage/assets/MASTER2_BASKET1.csv", index_col=0, parse_dates=True)
basket1_pr["mcap"] = b1_base["price_return"]
basket1_tr["mcap"] = b1_base["total_return"]

b2_base = pd.read_csv("/home/sronly/sr-storage/assets/MASTER2_BASKET2.csv", index_col=0, parse_dates=True)
basket2_pr["mcap"] = b2_base["price_return"]
basket2_tr["mcap"] = b2_base["total_return"]

basket1_pr = pd.DataFrame(basket1_pr).loc[:, ["mcap"] + ACTs]
basket1_tr = pd.DataFrame(basket1_tr).loc[:, ["mcap"] + ACTs]
basket2_pr = pd.DataFrame(basket2_pr).loc[:, ["mcap"] + ACTs]
basket2_tr = pd.DataFrame(basket2_tr).loc[:, ["mcap"] + ACTs]

basket1_pr.to_csv(DST_DIR / "basket1_pr.csv")
basket1_tr.to_csv(DST_DIR / "basket1_tr.csv")
basket2_pr.to_csv(DST_DIR / "basket2_pr.csv")
basket2_tr.to_csv(DST_DIR / "basket2_tr.csv")
