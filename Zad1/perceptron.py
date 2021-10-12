from logger import Logger as log
import numpy as np


class Perceptron:
    def __init__(self, alpha, weights, bias, dynamic_bias=True):
        self.alpha = alpha
        self.weights = weights  # bias is w0
        self._dynamic_bias = dynamic_bias
        self.weights = [bias] + self.weights
        self.iteration_counter = 0

    def _activation_func(self, x):
        bias = self.weights[0] if self._dynamic_bias else 0
        return np.dot(x, self.weights[1:]) + bias

    def _output_function(self, z):
        if not self._dynamic_bias:
            return (z > self.weights[0]) * 1
        return (z > 0) * 1

    @staticmethod
    def _get_error(expected, actual):
        return expected - actual

    def _update_weights(self, error, input_data):
        if self._dynamic_bias:
            self.weights[0] += self.alpha * error
        self.weights[1:] += self.alpha * error * input_data

    def train(self, train_data, train_output):
        no_errors = False
        while not no_errors:
            no_errors = True
            self.iteration_counter += 1
            num_of_errors = 0
            for x, y in zip(train_data, train_output):
                z = self._activation_func(x)
                perceptron_output = self._output_function(z)
                perceptron_error = self._get_error(
                    expected=y, actual=perceptron_output)
                self._update_weights(error=perceptron_error, input_data=x)
                if perceptron_error != 0:
                    no_errors = False
                    num_of_errors += 1
        log.info(
            f"Training finished in {self.iteration_counter} iterations")
        log.info(f"Bias: {self.weights[0]}")
        log.info(f"Weights: {self.weights[1:]}")

    def predict(self, input_vector):
        z = self._activation_func(input_vector)
        return self._output_function(z)


class BiPolarPerceptron(Perceptron):

    def __init__(self, alpha, weights, bias, dynamic_bias=True):
        super().__init__(alpha, weights, bias, dynamic_bias=dynamic_bias)

    def _output_function(self, z):
        if not self._dynamic_bias:
            return 1 if (z > self.weights[0]) else -1
        return 1 if z > 0 else -1
