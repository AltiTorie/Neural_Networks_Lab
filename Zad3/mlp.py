import numpy as np
from numpy.lib.function_base import average
from math import floor
import utils


class MLP:
    def __init__(self, starting_neurons, layers, labels, bias, learning_factor) -> None:
        self.neurons = [starting_neurons]
        for i in range(layers):
            self.neurons.append(floor(average([self.neurons[-1], labels])))
        self.neurons.append(labels)
        weights = []
        for l in range(layers + 1):
            weights.append(np.random.normal(0, 0.5, (self.neurons[l]+1, self.neurons[l+1])))
        self.layers = layers + 1
        self.bias = bias
        self.labels = labels
        self.weights = weights
        self.learning_factor = learning_factor

    @utils.time_usage
    def train(self, training_data, training_labels):
        for data, labels in zip(training_data, training_labels):
            a = data.flatten()
            for layer in range(self.layers):
                a = np.insert(a, 0, 1, axis=0)
                z = self.__full_excitation(a, self.weights[layer])
                a = list(map(self.__activation_functtion, z))

            o = self.__output_function(a)
            # update weights
        return o

    def predict(self):
        pass

    def __output_function(self, Z):
        S = sum([np.exp(z) for z in Z])
        return [np.exp(z)/S for z in Z]

    def __activation_functtion(self, z):
        a = 1 / (1 + np.exp(-z))
        return a

    def __full_excitation(self, X, weights):
        return np.dot(X, weights)
