from Utilities.logger import Logger as log
from copy import copy


class Adaline:

    def __init__(self, ni, weights, bias, accepted_epsilon) -> None:
        self.ni = ni
        self.weights = [bias] + weights
        self.accepted_epsilon = accepted_epsilon
        self.iteration_counter = 0

    def train(self, train_data, train_output):
        epsilon = self.accepted_epsilon + 0.1
        L = len(train_data)
        while epsilon > self.accepted_epsilon:
            self.iteration_counter += 1
            epsilons = []
            for xi, d in zip(train_data, train_output):
                y = 0
                for xk, w in zip(xi, self.weights[1:]):
                    y += xk*w
                y += self.weights[0]
                delta = (d - y)
                epsilons.append(delta**2)
                self._update_weights(delta=delta, x=xi)
            epsilon = (1 / L) * sum(epsilons)
            print(epsilon)
        log.info(f"Training finished in {self.iteration_counter} iterations")
        log.info(f"Reached epsilon: {epsilon}")
        log.info(f"Bias: {self.weights[0]}")
        log.info(f"Weights: {self.weights[1:]}")

    def _update_weights(self, delta, x):
        nw = [self.weights[0] + self.ni * delta]
        for xi, w in zip(x, self.weights[1:]):
            nw.append(w + (self.ni * delta * xi))
        self.weights = copy(nw)

    def predict(self, X):
        out = 0
        for x, w in zip(X, self.weights[1:]):
            out += x*w
        return self._bipolar_function(out + self.weights[0])

    def _bipolar_function(self, z):
        return 1 if z > 0 else -1
