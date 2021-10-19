import numpy as np
from numpy.lib.function_base import average
from math import floor


class MLP:
    def __init__(self, starting_neurons, training_data_size, layers, labels, bias, learning_factor) -> None:
        self.neurons = [floor(average([starting_neurons, labels]))]
        for i in range(layers):
            self.neurons.append(floor(average([self.neurons[-1], labels])))
        self.neurons.append(labels)
        print(self.neurons)

        weights = [np.random.rand(starting_neurons, training_data_size)]
        for l in range(layers):
            weights.append(np.random.rand(self.neurons[l], self.neurons[l+1]))
        self.layers = layers
        self.bias = bias
        self.labels = labels
        self.weights = weights
        self.learning_factor = learning_factor

    def train(self, training_data, training_labels):
        z = self.__full_excitation(training_data, self.weights[0])
        a = self.__activation_functtion(z)
        for layer in range(self.layers):
            for neuron in range(self.neurons[layer]):
                zl = self.__full_excitation(z, self.weights[layer+1][neuron])
                al = self.__activation_functtion(zl)
        return al

    def predict(self):
        pass

    def __output_function(Z):
        S = sum([np.exp(z) for z in Z])
        return [np.exp(z)/S for z in Z]

    def __activation_functtion(self, z):
        a = 1 / (1 + np.exp(z))
        return a

    def __full_excitation(self, X, weights):
        S = sum([x * w for (x, w) in zip(X.flatten(), weights[1:])])
        S += weights[0]
        return S
