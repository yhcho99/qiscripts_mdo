import sys
import os.path
from pathlib import Path
import subprocess

python_path = sys.executable
script_path = os.path.join("product", "master", "master_after_220301.py")
cwd = str(Path(".").absolute())


training_universe = ["it", "nasdaq"]
input_factors = ["five_factors", "qraft_factors"]
year_list = [1, 3, 5]

for universe in training_universe:
    for factors in input_factors:
        for year in year_list:
            done = False
            while not done:
                proc = subprocess.run([python_path, script_path, universe, factors, str(year)], cwd=cwd)
                if isinstance(proc, subprocess.CompletedProcess):
                    if proc.returncode == 0:
                        done = True
                    elif proc.returncode == 1:
                        done = False
