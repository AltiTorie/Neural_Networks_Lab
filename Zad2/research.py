from logger import Logger as log
import random
from adaline import Adaline
# from perceptron import Perceptron, BiPolarPerceptron
import numpy as np
import utils
import copy
import datetime


@utils.time_usage
def train_adaline(adaline: Adaline, learning_data, output_data):
    adaline.train(learning_data, output_data)


def task_1():
    # dynamic bias, different starting weights
    # TESTING: Starting weights range ex. (-1, 1), (-0.2, 0.2)
    print("Task 2 running")
    WEIGHTS_RANGES = [[[-1, -0.8], [0.8, 1]],
                      [[-0.8, -0.5], [0.5, 0.8]],
                      [[-0.5, -0.25], [0.25, 0.5]],
                      [[-0.25, -0.1], [0.1, 0.25]],
                      [[-0.1, -0.01], [0.01, 0.1]],
                      [[-0.01, 0.000001], [0.000001, 0.01]],
                      [[-0.000001, -0.0000001], [0.0000001, 0.000001]]]
    weights = [[random.choice([random.uniform(wr[0][0], wr[0][1]), random.uniform(wr[1][0], wr[1][1])]),
                random.choice([random.uniform(wr[0][0], wr[0][1]), random.uniform(wr[1][0], wr[1][1])])]
               for wr in WEIGHTS_RANGES]
    ni = 0.05
    bias = 0.1
    LEARNING_DATA, EXPECTED = utils.generate_random_points(10)
    EXPECTED = [-1 if e == 0 else 1 for e in EXPECTED]
    EPSILON = 0.3
    TEST_POINTS = [[0, 0], [0, 1], [1, 0], [1, 1], [0.0111, -0.0111],
                   [1.001, 0.0111], [-0.0111, 0.9789], [0.987, 1.0123]]
    TEST_EXPECTED = [-1, -1, -1, 1, -1, -1, -1, 1]
    log.info(f"Testing weights ranges - starting data:")
    log.info(f"ni: {ni}")
    log.info(f"weights: {weights}")
    log.info(f"bias: {bias}")
    log.info(f"accepted epsilon: {EPSILON}")
    log.info(f"__________________________")
    for w in weights:
        a = Adaline(ni=ni, weights=copy.copy(w),
                    bias=bias, accepted_epsilon=EPSILON)
        log.info(f"Training weigts: {w}")
        train_adaline(a, LEARNING_DATA, EXPECTED)
        counter = 0
        for test, expected in zip(TEST_POINTS, TEST_EXPECTED):
            pred = a.predict(test)
            counter += pred == expected
        log.info(f"Weights {w} - accuracy: {counter}/{len(TEST_POINTS)}")
        log.info("_______________________________")

# def task__():
#     # small starting weights, static threshold, different thresholds.
#     # TESTING: Different static thresholds

#     print("Task 1 running")
#     weights = [round(random.uniform(-0.01, 0.01), 3),
#                round(random.uniform(-0.01, 0.01), 3)]
#     alpha = 0.05
#     thresholds = [0.01, 0.1, 0.3, 0.5, 1, 2, 5, 10, 25, 50, 100, 1000]
#     LEARNING_DATA, EXPECTED = utils.generate_random_points(10)
#     TEST_POINTS = [[0, 0], [0, 1], [1, 0], [1, 1], [
#         0.0111, -0.0111], [1.001, 0.0111], [-0.0111, 0.9789], [0.987, 1.0123]]
#     TEST_EXPECTED = [0, 0, 0, 1, 0, 0, 0, 1]

#     log.info(f"Testing different static thresholds - starting data:")
#     log.info(f"alpha: {alpha}")
#     log.info(f"weights: {weights}")
#     log.info(f"thresholds: {thresholds}")
#     log.info(f"__________________________")
#     for threshold in thresholds:
#         p = Perceptron(alpha=alpha, weights=copy.copy(
#             weights), bias=threshold, dynamic_bias=False)
#         log.info(f"Training threshold: {threshold}")
#         train_perceptron(p, LEARNING_DATA, EXPECTED)
#         counter = 0
#         for test, expected in zip(TEST_POINTS, TEST_EXPECTED):
#             pred = p.predict(test)
#             counter += pred == expected
#         log.info(
#             f"Threshold {threshold} - accuracy: {counter}/{len(TEST_POINTS)}")
#         log.info("_______________________________")


def task_2():
    # dynamic bias, small starting weights, different alpha
    # TESTING: learning factor ni
    print("Task 3 running")
    weights = [round(random.uniform(-0.01, 0.01), 3),
               round(random.uniform(-0.01, 0.01), 3)]
    # nis = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
    nis = [0.999]
    bias = 0.01
    EPSILON = 0.1
    LEARNING_DATA, EXPECTED = utils.generate_random_points(1)
    TEST_POINTS = [[0, 0], [0, 1], [1, 0], [1, 1], [
        0.0111, -0.0111], [1.001, 0.0111], [-0.0111, 0.9789], [0.987, 1.0123]]
    TEST_EXPECTED = [-1, -1, -1, 1, -1, -1, -1, 1]
    log.info(f"Testing learning factor alpha - starting data:")
    log.info(f"nis: {nis}")
    log.info(f"weights: {weights}")
    log.info(f"bias: {bias}")
    log.info(f"__________________________")
    for ni in nis:
        a = Adaline(ni=ni, weights=copy.copy(weights),
                    bias=bias, accepted_epsilon=EPSILON)
        log.info(f"Training alpha: {ni}")
        train_adaline(a, LEARNING_DATA, EXPECTED)
        counter = 0
        for test, expected in zip(TEST_POINTS, TEST_EXPECTED):
            pred = a.predict(test)
            counter += pred == expected
        log.info(f"Ni {ni} - accuracy: {counter}/{len(TEST_POINTS)}")
        log.info("_______________________________")


# def task_4():
#     # dynamic bias, small starting weights, different activation function
#     # TESTING: difference between unipolar and bipolar activation functions
#     print("Task 4 running")
#     weights = [round(random.uniform(-0.01, 0.01), 3),
#                round(random.uniform(-0.01, 0.01), 3)]
#     alpha = 0.0001
#     bias = 0.0001
#     LEARNING_DATA, EXPECTED = utils.generate_random_points(1)
#     TEST_POINTS = [[0, 0], [0, 1], [1, 0], [1, 1], [
#         0.0111, -0.0111], [1.001, 0.0111], [-0.0111, 0.9789], [0.987, 1.0123]]
#     TEST_EXPECTED = [0, 0, 0, 1, 0, 0, 0, 1]
#     log.info(f"Testing difference between unipolar and bipolar - starting data:")
#     log.info(f"alphas: {alpha}")
#     log.info(f"weights: {weights}")
#     log.info(f"bias: {bias}")
#     log.info(f"__________________________")

#     p = Perceptron(alpha=alpha, weights=copy.copy(
#         weights), bias=bias, dynamic_bias=True)
#     log.info(f"Training perceptron: {p.__class__.__name__}")
#     train_perceptron(p, LEARNING_DATA, EXPECTED)
#     counter = 0
#     for test, expected in zip(TEST_POINTS, TEST_EXPECTED):
#         pred = p.predict(test)
#         counter += pred == expected
#     log.info(f"Output perceptron {p.__class__.__name__} - accuracy: {counter}/{len(TEST_POINTS)}")
#     log.info("_______________________________")

#     p = BiPolarPerceptron(alpha=alpha, weights=copy.copy(
#         weights), bias=bias, dynamic_bias=True)
#     log.info(f"Training perceptron: {p.__class__.__name__}")
#     EXPECTED_BIPOLAR = [-1 if e == 0 else 1 for e in EXPECTED]
#     TEST_EXPECTED_BIPOLAR = [-1 if e == 0 else 1 for e in TEST_EXPECTED]
#     train_perceptron(p, LEARNING_DATA, EXPECTED_BIPOLAR)
#     counter = 0
#     for test, expected in zip(TEST_POINTS, TEST_EXPECTED_BIPOLAR):
#         pred = p.predict(test)
#         counter += pred == expected
#     log.info(
#         f"Output perceptron {p.__class__.__name__} - accuracy: {counter}/{len(TEST_POINTS)}")
#     log.info("_______________________________")


if __name__ == "__main__":
    log.info(f"_____________________ {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')} _____________________")
    # task_1()
    for i in range(1):
        # task_1()
        task_2()
    # task_4()
