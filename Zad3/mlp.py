import numpy as np
from numpy.lib.function_base import average
from math import floor
import utils


class MLP:
    def __init__(self, starting_neurons, layers, labels, bias, learning_factor) -> None:
        self.neurons = [starting_neurons]
        for _ in range(layers):
            self.neurons.append(floor(average([self.neurons[-1], labels])))
        self.neurons.append(labels)
        weights = []
        for l in range(layers + 1):
            weights.append(np.random.normal(0, 0.5, (self.neurons[l]+1, self.neurons[l+1])))
        print(len(weights[3][0]))
        self.layers = layers + 1
        self.bias = bias
        self.labels = labels
        self.weights = weights
        self.learning_factor = learning_factor

    @utils.time_usage
    def train(self, training_data, training_labels):
        for data, label in zip(training_data, training_labels):
            a = data.flatten()
            a = np.insert(a, 0, 1, axis=0)
            A = [a]
            for layer in range(self.layers):
                z = self.__full_excitation(a, self.weights[layer])
                a = list(map(self.__activation_function, z))
                if layer != self.layers-1:
                    a = np.insert(a, 0, 1, axis=0)
                A.append(a)
            o = self.__soft_max(a)
            # predicted = o.index(max(o))
            label_vector = np.zeros(self.labels)
            label_vector[label] = 1
            cost = self.__cost_function(o, label_vector, A[-1])
            delta = self.calculate_output_delta(cost, np.insert(list(map(self.__activation_function_der, z))))
            deltas = [delta]
            for layer in range(self.layers-1, 0, -1):
                print(f"W: {self.weights[layer].shape} | D: {len(deltas[-1])}")
                dd = ((self.weights[layer] @ deltas[-1]))
                print(f"D: {len(dd)} | A: {len(A[layer])}")
                dlx = (dd * A[layer])
                print(f"dlx: {len(dlx)}")
                deltas.append(dlx)
            for layer in range(self.layers-1, 0, -1):
                self.weights[layer] = self.weights[layer] - \
                    ((self.learning_factor / len(training_data)) * deltas[layer] @ A[layer-1].T)
        return o

    def softmax_gradient(self, y, a):
        return -(y - a)

    def calculate_output_delta(self, C, fz):
        return C*fz

    def __cost_function(self, y, expected, a):
        q = -(expected - y)
        print(f"QS: {q.shape}")
        print(f"ATS: {np.transpose(a).shape}")
        return q*(np.transpose(a))
        # return (y - expected)/(y*(1-y))

    def predict(self, data):
        a = data.flatten()
        for layer in range(self.layers):
            a = np.insert(a, 0, 1, axis=0)  # insert bias=1 at 0 index
            z = self.__full_excitation(a, self.weights[layer])
            a = list(map(self.__activation_function, z))
        o = self.__soft_max(a)
        o = o.index(max(o))
        return(o)

    def __soft_max(self, Z):
        S = sum([np.exp(z) for z in Z])
        return [np.exp(z)/S for z in Z]

    def __activation_function(self, z):
        a = 1 / (1 + np.exp(-z))
        return a

    def __activation_function_der(self, z):
        return (1 - self.__activation_function(z)) * self.__activation_function(z)

    def __full_excitation(self, X, weights):
        print(X.shape)
        print(weights.shape)
        return X @ weights
