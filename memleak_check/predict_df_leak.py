#%%
from ctypes import alignment
import gc
import os
import sys
import time

from pandas.core.dtypes import dtypes

import catboost as cb
import pandas as pd
import numpy as np
import psutil

clock = time.time()
maxmem = 0.0

def memory_footprint(it: int, prev_snapshot=None):
    """Returns memory (in MB) being used by Python process"""
    # gc.collect()
    mem = psutil.Process(os.getpid()).memory_info().rss
    return mem / 1024 ** 2, None  # snapshot


def X(batch_size: int):
    for _ in range(batch_size):
        yield [
            "1", "2", "3", 0.33, 0.5
        ]


def main(batch_size=15, n_iterations=100, print_every=10, cleanup_every=None):
    global clock, maxmem

    print("python version=", sys.version)
    print("numpy version=", np.__version__)
    print("catboost version=", cb.__version__)

    model = cb.CatBoost()
    model.load_model(fname="model.cbm")
    
    snapshot = None

    # tracemalloc.start(10)
    for i in range(n_iterations + 1):
        data = pd.DataFrame(
            [x for x in X(batch_size)],
            columns=["a", "b", "c", "d", "e"]
        )
        # Predict directly without creating a Pool
        y = model.predict(data, thread_count=1)

        if i and i % print_every == 0:
            mem, snapshot = memory_footprint(i, snapshot)
            maxmem = max(mem, maxmem)
            elapsed = time.time() - clock
            print("iter {} y={} elapsed={:.3f}s ms/call: {:.6f} ms/item: {:.6f} mem: {:.3f}Mb max: {:.3f}Mb".format(
                i, y[0], elapsed, 1000. * elapsed / print_every, 1000. * elapsed / batch_size / print_every, mem, maxmem
            ))
            sys.stdout.flush()
            clock = time.time()
# %%

if __name__ == "__main__":
    main(batch_size=150, n_iterations=3000000, print_every=1000)
# %%
