import numpy as np
import random
import time
from logger import Logger as log

def time_usage(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        log.info(f"Time spent on {func.__name__}: {time.time() - start}")
        return output
    return wrapper

def generate_random_points(sets_count):
    data = []
    outputs = []
    for _ in range(sets_count):
        # points around (0,0)
        data.append(
            np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]))
        outputs.append(0)
        # points around (0, 1)
        data.append(
            np.array([random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1)]))
        outputs.append(0)
        # points around (1, 0)
        data.append(
            np.array([random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1)]))
        outputs.append(0)
        # points around (1, 1)
        data.append(
            np.array([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)]))
        outputs.append(1)
    return data, outputs