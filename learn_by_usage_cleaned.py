#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
based on: https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24
    and https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network/29889993
Created on Sun Jul 19 15:45:02 2020

@author: detlef
"""

import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
from matplotlib import pyplot
from math import cos, sin, atan


lr = 0.9
hidden_size = 4
float_mean = 0.01
scale_linewidth = 0.1

# input data
inputs = np.array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 0],
                   [1, 1, 1]])
# output data
outputs = np.array([[0], [0], [1], [0], [1], [1], [1], [1]])

np.seterr(under='ignore', over='ignore')

def sigmoid(x):
    xx = x
    return 1 / (1 + np.exp(-xx))

def sigmoid_derivative(x):
    xx = x
    return (np.exp(-xx) / (np.exp(-xx) + 1) ** 2) #x * (1 - x) # this was an optimized derivative using the self.hidden

def transform_01_mp(x):
    return 2*x - 1

vertical_distance_between_layers = 6
horizontal_distance_between_neurons = 2
neuron_radius = 0.5
number_of_neurons_in_widest_layer = 4
class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, v):
        if v > 0:
            circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False, color='green')
        else:
            circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False, color='gray')
        pyplot.gca().add_patch(circle)

class Layer():
    def __init__(self, network, number_of_neurons, weights, values):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights
        if weights is not None:
            self.stats = np.zeros(weights.shape)
        self.values = values

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth, graylevel = None):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        if linewidth > 0:
            c = 'green'
        else:
            c = 'red'
        if graylevel is not None:
            c = (1-graylevel, 1-graylevel, 1-graylevel)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=np.abs(linewidth * scale_linewidth), color = c)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw(round(self.values[this_layer_neuron_index]))
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[previous_layer_neuron_index, this_layer_neuron_index]
                    stability = self.previous_layer.stats[previous_layer_neuron_index, this_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, 40, stability)
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)
                    
    def backward(self, post_layer, post_error):
        pre_error = np.dot(post_error * sigmoid_derivative(post_layer), self.weights.T)
        d_weights = np.dot(self.values.T, (2 * post_error * sigmoid_derivative(post_layer)))
        
        self.change_weights(d_weights)
        return pre_error # first idea to the layer backpropergation
    
    def forward(self, pre_layer, dostats):
        self.values = pre_layer
        if self.weights is None:
            return pre_layer
        post_layer = sigmoid(np.dot(pre_layer, self.weights))
        if dostats:
            post_l = transform_01_mp(np.expand_dims(post_layer,-2))
            pre_l = transform_01_mp(np.expand_dims(pre_layer, -2))
            #print(np.transpose(post_l[2]), pre_l[2])
            stability = np.matmul(pre_l.swapaxes(-1,-2), post_l)*np.tanh(self.weights)
            if len(stability.shape) == 2:
                stability = np.expand_dims(stability, 0) # handle single and multi inputs
            stability = np.sum(stability, axis = 0) / len(stability)
            print(stability)
            self.stats = float_mean * stability + (1-float_mean) * self.stats
        return post_layer
        
    def change_weights(self, d_weights):
        self.weights += d_weights * lr
    
class DrawNet():
    def __init__(self):
        self.layers = []
        self.epoch_list = []
        self.error_history = []
        
    def add_layer(self, number_of_neurons, weights=None, values=None):
        layer = Layer(self, number_of_neurons, weights, values)
        self.layers.append(layer)
    
    def forward(self, dostats = False):
        outp = self.layers[0].values
        for layer in self.layers:
            outp = layer.forward(outp, dostats)
        #self.layers[-1].values = outp
        return outp
    
    def backward(self):
        self.error = pre_error = self.y - self.layers[-1].values
        for i in reversed(range(len(self.layers)-1)):
            pre_error = self.layers[i].backward(self.layers[i+1].values, pre_error)
        return pre_error
    
    def train(self, epochs=1000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.forward(True)
            # go back though the network to make corrections based on the output
            self.backward()    
            # keep track of the error history over each epoch
            self.error_history.append(np.sum(np.square(self.error)))
            self.epoch_list.append(epoch)

    
    def set_input(self, new_input, new_output):
        self.layers[0].values = new_input
        self.y = new_output
        
    def draw(self, result):
        for layer in self.layers:
            layer.draw()
        if result is not None:
            if result[0] > 0:
                circle = pyplot.Circle((self.layers[-1].neurons[0].x, self.layers[-1].neurons[0].y), radius=neuron_radius+0.3, fill=False, color='green')
            else:
                circle = pyplot.Circle((self.layers[-1].neurons[0].x, self.layers[-1].neurons[0].y), radius=neuron_radius+0.3, fill=False, color='gray')
            pyplot.gca().add_patch(circle)
        pyplot.axis('scaled')
        pyplot.show(dpi=1200)
        
    def predict(self, new_input, oo = None, drawit=False):
        self.set_input(new_input, oo)
        prediction = self.forward(True)
        if drawit:
            self.draw(oo)
        return prediction
        
NN2 = DrawNet()
NN2.add_layer(3, np.random.rand(inputs.shape[1], hidden_size), None)
NN2.add_layer(hidden_size, np.random.rand(hidden_size, 1), None)
NN2.add_layer(1, None, None)
NN2.set_input(inputs, outputs)

# train neural network
#NN2.train()


#testing single inputs for few shot learning
error_history = []
epoch_list = []
for epoch in range(1000):
    for i in range(len(inputs)):
        NN2.set_input(inputs[i:i+1], outputs[i:i+1])
        NN2.forward(True)
        NN2.backward()
        error_history.append(sum(np.square(NN2.error)))
        epoch_list.append(epoch)

for i in range(len(inputs)):
    print(NN2.predict(inputs[i], outputs[i], drawit=True), 'correct', outputs[i])

# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(epoch_list, error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

print('Error', error_history[-1])
print(NN2.layers[0].stats, NN2.layers[1].stats)
