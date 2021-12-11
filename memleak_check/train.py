#%%
import gc
import os
import sys
import random

import catboost as cb
import numpy as np
import psutil

import tracemalloc


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

def Y(batch_size: int):
    for _ in range(batch_size):
        yield random.random()


def main(batch_size=15, n_iterations=100, print_every=10, cleanup_every=None):
    print("python version=", sys.version)
    print("numpy version=", np.__version__)
    print("catboost version=", cb.__version__)

    data = np.asarray([x for x in X(batch_size)], dtype=object)
    labels = np.asarray([y for y in Y(batch_size)])

    print(type(data), isinstance(data, np.ndarray))

    model = cb.CatBoost()

    for i in range(n_iterations):

        model.fit(data, labels, cat_features=[0, 1, 2])
        if i % print_every == 0:
            mem, snapshot = memory_footprint(i)
            print("Memory usage (iter {}): {:.2f} MB".format(i, mem))

    model.save_model("model.cbm")
# %%

if __name__ == "__main__":
    main(batch_size=15000, n_iterations=10, print_every=10, cleanup_every=500)
# %%
