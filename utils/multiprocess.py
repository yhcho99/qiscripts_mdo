import time
import random
import multiprocessing as mp


class MultiProcess:
    def __init__(self):
        self.proc_list = []

    def cpu_count(self, max_count=None):
        if max_count is not None:
            self.cpu_count = max_count
        else:
            self.cpu_count = int(mp.cpu_count()/2)

    def run_process(self, proc):
        self.proc_list.append(proc)
        if len(self.proc_list) >= self.cpu_count:
            while True:
                time.sleep(0.1)
                for i, p in enumerate(self.proc_list):
                    if not p.is_alive():
                        self.proc_list.pop(i)
                        return None

    def final_process(self):
        self.proc_list = [proc.join() for proc in self.proc_list]
