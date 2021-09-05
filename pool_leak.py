#%%
import gc
import os
import random
import sys
import time

import catboost as cb
# import fastpool as fp  # replacement for catboost python wrapper
import numpy as np
import psutil
import ctypes

import tracemalloc

clock = time.time()
maxmem = 0.0

def memory_footprint(it: int, prev_snapshot=None):
    """Returns memory (in MB) being used by Python process"""
    gc.collect()
    mem = psutil.Process(os.getpid()).memory_info().rss
    return mem / 1024 ** 2, None  # snapshot


def X(batch_size: int):
    for _ in range(batch_size):
        yield [
            random.randint(0, 9), random.randint(0, 9), random.randint(0, 9), random.random(), random.random()
        ]
        

def main(batch_size=15, n_iterations=100, print_every=10, cleanup_every=None):
    global clock, maxmem

    print("python version=", sys.version)
    print("numpy version=", np.__version__)
    print("catboost version=", cb.__version__)

    model = cb.CatBoost()
    model.load_model(fname="model.cbm")
    
    # data = np.asarray([x for x in X(batch_size)], dtype=object)
    snapshot = None

    # tracemalloc.start(10)
    for i in range(n_iterations):
        pool = cb.Pool([x for x in X(batch_size)], cat_features=[0, 1, 2], thread_count=1)
        y = model.predict(pool, thread_count=1)

        if i % print_every == 0:
            mem, snapshot = memory_footprint(i, snapshot)
            maxmem = max(mem, maxmem)
            elapsed = time.time() - clock
            print("iter {} y={} elapsed={:.3f}s ms/call: {:.6f} ms/item: {:.6f} mem: {:.3f}Mb max: {:.3f}Mb".format(
                i, y[0], elapsed, 1000. * elapsed / print_every, 1000. * elapsed / batch_size / print_every, mem, maxmem
            ))
            clock = time.time()

        # print(pool)
        # ref = ctypes.py_object(pool)
        # ctypes.resize(ref, 8)
        # del ref
        # del pool
        # ctypes.pythonapi._Py_Dealloc(ref)

        # del pool

# %%

if __name__ == "__main__":
    main(batch_size=15, n_iterations=1000000, print_every=1000)
# %%
