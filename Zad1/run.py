import random
from logger import Logger as log
import numpy as np
from perceptron import Perceptron
import utils

if __name__ == "__main__":
    weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
    margin = random.uniform(0.1, 1)
    threshold = [random.uniform(0.1, 1) for _ in range(10)]
    LEARNING_FACTOR = 0.01
    LEARNING_DATA, EXPECTED = utils.generate_random_points(4)
    p = Perceptron(LEARNING_FACTOR, weights, margin, False)
    p.train(LEARNING_DATA, EXPECTED)
    log.info("Testing perceptron")
    log.info(f"[0, 0] -> {p.predict(np.array([0, 0]))}")
    log.info(f"[0, 1] -> {p.predict(np.array([0, 1]))}")
    log.info(f"[1, 0] -> {p.predict(np.array([1, 0]))}")
    log.info(f"[1, 1] -> {p.predict(np.array([1, 1]))}")

    log.info(f"[0.0123, -0.0456] -> {p.predict(np.array([0.0123, -0.0456]))}")
    log.info(f"[1.0123, -0.0456] -> {p.predict(np.array([1.0123, -0.0456]))}")
    log.info(f"[0.0123, 1.0456] -> {p.predict(np.array([0.0123, 1.0456]))}")
    log.info(f"[1.0123, 1.0456] -> {p.predict(np.array([1.0123, 1.0456]))}")
