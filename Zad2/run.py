import random
from adaline import Adaline
from logger import Logger as log
import utils

if __name__ == "__main__":
    LEARNING_FACTOR = 0.001
    ACCEPTED = 0.3
    BIAS = 0.1
    X, D = utils.generate_random_points(10)
    D = [-1 if d == 0 else 1 for d in D]
    weights = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]
    adaline = Adaline(ni=LEARNING_FACTOR, weights=weights,
                      bias=BIAS, accepted_epsilon=ACCEPTED)
    adaline.train(X, D)

    log.info("Testing Adaline")
    log.info(f"[0, 0] -> {adaline.predict([0, 0])}")
    log.info(f"[0, 1] -> {adaline.predict([0, 1])}")
    log.info(f"[1, 0] -> {adaline.predict([1, 0])}")
    log.info(f"[1, 1] -> {adaline.predict([1, 1])}")

    log.info(f"[0.0123, -0.0456] -> {adaline.predict([0.0123, -0.0456])}")
    log.info(f"[1.0123, -0.0456] -> {adaline.predict([1.0123, -0.0456])}")
    log.info(f"[0.0123, 1.0456] -> {adaline.predict([0.0123, 1.0456])}")
    log.info(f"[1.0123, 1.0456] -> {adaline.predict([1.0123, 1.0456])}")
