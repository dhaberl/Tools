from multiprocessing import Pool, cpu_count
from time import perf_counter, sleep

import numpy as np


def prepare_mp_pool(x_vals, y_vals):
    """
    This function prepares the multiprocessing pool.
    It shall return a list of iterable arguments which
    will serve as input for the my_func_to_parallelize().
    """
    iterable_args = []
    for x, y in zip(x_vals, y_vals):
        iterable_args.append((x, y))

    return iterable_args


def my_func_to_parallelize(x, y):
    """
    This is the function which should run in parallel on multiple cores.
    Put your code here.
    """
    z = x + y

    sleep(1)  # simulates long calculation time

    return z


if __name__ == "__main__":

    start = perf_counter()

    num_cpus = cpu_count()
    print(f"Available number of CPU core(s): {num_cpus}")

    n_jobs = 2
    print(f"Using {n_jobs} CPU core(s)")

    # Calculate pairwise sum of x and y
    x_vals = np.arange(0, 10)
    y_vals = np.arange(10, 20)

    # Prepare multiprocessing pool
    iterable_args = prepare_mp_pool(x_vals, y_vals)

    # Multiprocessing
    with Pool(processes=n_jobs) as p:
        xy_sum = p.starmap(my_func_to_parallelize, iterable_args)

    print(xy_sum)

    end = perf_counter()
    delta = end - start
    print(f"{n_jobs} core(s) used. Took {delta:.2f} seconds.")
