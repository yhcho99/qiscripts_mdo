from pathlib import Path
import os
from shutil import copy2
import subprocess


TARGET = "ff+wo_alt"
TRY = 1
ACTs = [0.15, 0.3, 0.5, 0.7, 0.99]

base_path = Path(f"/home/sronly/sr-storage/master_phase2-{TARGET}-try{TRY}")
basket1_base_path = base_path / "infer"
basket1_base_filelist = os.listdir(basket1_base_path)

basket2_base_path = base_path / "intermediate" / "BASKET2" / "infer"
basket2_base_filelist = os.listdir(basket2_base_path)

f = open(base_path / "infer.txt", "a+")

for act in ACTs:
    f.write(f"ACT{act} Try{TRY} Start\n")

    target_path = Path(f"/home/sronly/sr-storage/master_phase2-{TARGET}-try{TRY}-ACT_CF{act:.4f}")

    basket1_target_path = target_path / "infer"
    basket1_target_path.mkdir(parents=True, exist_ok=True)
    basket1_target_filelist = os.listdir(basket1_target_path)

    f.write("Basket1 Infer Copy Start\n")
    for fp in basket1_base_filelist:
        if fp[-4:] != ".csv":
            continue

        if fp in basket1_target_filelist:
            continue
        
        src_fp = (basket1_base_path / fp).absolute().as_posix()
        dst_fp = (basket1_target_path / fp).absolute().as_posix()
        
        copy2(src_fp, dst_fp)
    f.write("Basket1 Infer Copy Done\n")

    basket2_target_path = target_path / "intermediate" / "BASKET2" / "infer"
    basket2_target_path.mkdir(parents=True, exist_ok=True)
    basket2_target_filelist = os.listdir(basket2_target_path)

    f.write("Basket2 Infer Copy Start\n")
    for fp in basket2_base_filelist:
        if fp[-4:] != ".csv":
            continue

        if fp in basket2_target_filelist:
            continue

        src_fp = (basket2_base_path / fp).absolute().as_posix()
        dst_fp = (basket2_target_path / fp).absolute().as_posix()

        copy2(src_fp, dst_fp)
    f.write("Basket2 Infer Copy Done\n")

    f.write("Infer Start\n")
    subprocess.run(
        [
            "/home/sronly/miniconda3/envs/etf/bin/python3.8",  
            f"experiment/master_phase2/pretest/inferscript.py",
            "--target", TARGET,
            "--act", f"{act:.4f}",
            "--tryv", f"{TRY}",
        ],
        cwd="/home/sronly/Projects/qiscripts",
    )
    f.write("Infer Done\n")
