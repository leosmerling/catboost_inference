#%%
import gc
import os
import random
import sys
import time

import catboost as cb
import fastpool as fp
import numpy as np
import psutil

import tracemalloc

clock = time.time()
maxmem = 0.0

def memory_footprint(it: int, prev_snapshot=None):
    """Returns memory (in MB) being used by Python process"""
    gc.collect()
    
    # snapshot = tracemalloc.take_snapshot()
    # if prev_snapshot:
    #     top_stats = snapshot.compare_to(prev_snapshot, 'lineno')
    # else:
    #     top_stats = snapshot.statistics('lineno')

    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)

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

    # features = [["X", "1", "0", 0.5, 0.33]] * batch_size
    # cat_indices = [0, 1, 2]

    model = fp.CatBoost()
    model.load_model(fname="model.cbm")
    

    # data = cb.FeaturesData(
    #     cat_feature_data=np.asarray([[2, 1, 0]] * batch_size, dtype=np.o),
    #     cat_feature_names=["A", "B", "C"],
    #     num_feature_data=np.asarray([[0.5, 0.33]] * batch_size, dtype=np.float32),
    #     num_feature_names=["D", "E"]
    # )

    #features = np.random.rand(batch_size, 5)
    #cat_indices = []

    data = np.asarray([x for x in X(batch_size)], dtype=object)

    snapshot = None
    # tracemalloc.start(10)
    for i in range(n_iterations):

        pool = fp.Pool(data, cat_features=[0, 1, 2], thread_count=1)
        y = model.predict(pool, thread_count=1)

        if i % print_every == 0:
            mem, snapshot = memory_footprint(i, snapshot)
            maxmem = max(mem, maxmem)
            elapsed = time.time() - clock
            print("iter {} y={} clock={:.3f} lat/item: {:.6f} mem: {:.2f}MB max: {:.2f}MB".format(
                i, y[0], elapsed, 1000. * elapsed / batch_size / print_every, mem, maxmem
            ))
            clock = time.time()
            del pool

# %%

if __name__ == "__main__":
    main(batch_size=15, n_iterations=1000000, print_every=1000)
# %%
