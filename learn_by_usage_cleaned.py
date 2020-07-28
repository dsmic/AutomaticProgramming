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

pyplot.rcParams['figure.dpi'] = 300

lr = 1
hidden_size = 3
stability_mean = 0.1
scale_linewidth = 0.1
weight_tanh_scale = 0.1
clip_weights = 10
scale_for_neuron_diff = 1
use_stability = False

scale_sigmoid = 3
shift_sigmoid = 1

few_shot_end = 0.3
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
outputs = np.array([[0], [1], [0], [0], [1], [1], [1], [1]])


np.seterr(under='ignore', over='ignore')

def sigmoid(x):
    xx = scale_sigmoid * (x - shift_sigmoid)
    return 1 / (1 + np.exp(-xx)) #* 2 -1

def sigmoid_derivative(x):
    xx = scale_sigmoid * (x - shift_sigmoid)
    return scale_sigmoid * (np.exp(-xx) / (np.exp(-xx) + 1) ** 2)

def transform_01_mp(x):
    return 2*x - 1


#inputs = transform_01_mp(inputs)
#outputs = transform_01_mp(outputs)




vertical_distance_between_layers = 6
horizontal_distance_between_neurons = 2
neuron_radius = 0.5
neuron_scale_line = 2.0
number_of_neurons_in_widest_layer = 4
class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, v):
        if v > 0:
            circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False, color='green', linewidth = 3)
        else:
            circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False, color='gray', linewidth = 3)
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

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth, graylevel = None, usage = None):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        if graylevel is None:
            nr = neuron_radius * neuron_scale_line
        else:
            nr = neuron_radius * 1.4
        x_adjustment = nr * sin(angle)
        y_adjustment = nr * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        if linewidth > 0:
            c = 'green'
        else:
            c = 'red'
        
        lw = linewidth * scale_linewidth
        if graylevel is not None:
            #graylevel = (graylevel +1)/2
            if graylevel < 0: graylevel = 0
            if graylevel > 1: graylevel = 1
            c = (0, 0, 1, graylevel)
            lw = linewidth
        if usage is not None:
            #graylevel = (graylevel +1)/2
            if usage < 0: usage = 0
            if usage > 1: usage = 1
            c = (1, 1- usage / 3, 1 - usage)
            lw = linewidth
            
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=np.abs(lw), color = c)
        pyplot.gca().add_line(line)

    def draw(self, usage):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw(round(self.values[this_layer_neuron_index]))
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[previous_layer_neuron_index, this_layer_neuron_index]
                    stability = self.previous_layer.stats[previous_layer_neuron_index, this_layer_neuron_index]
                    used = 0
                    if weight > 0:
                        used = self.previous_layer.values[previous_layer_neuron_index] * self.values[this_layer_neuron_index]
                    else:
                        used = self.previous_layer.values[previous_layer_neuron_index] * (1 - self.values[this_layer_neuron_index])
                                                              
                    print("connection %2d %2d    %6.3f    %6.3f    %6.3f    %6.3f used: %6.3f" % (previous_layer_neuron_index, this_layer_neuron_index, self.previous_layer.values[previous_layer_neuron_index], self.values[this_layer_neuron_index], weight, stability, used))
                    if usage:
                        self.__line_between_two_neurons(neuron, previous_layer_neuron, 4, usage = used)
                    else:
                        self.__line_between_two_neurons(neuron, previous_layer_neuron, 4, stability)
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
            post_l = np.expand_dims(post_layer,-2)
            pre_l_2d = np.expand_dims(pre_layer, -2)
            
            # this is necessary if 0 1 neurons are used, not if -1 1 ones
            post_l = transform_01_mp(post_l)
            pre_l = transform_01_mp(pre_l_2d)
            
            #print(np.transpose(post_l[2]), pre_l[2])
            stability = (np.tanh(scale_for_neuron_diff * np.matmul(pre_l.swapaxes(-1,-2), post_l)) * np.tanh(self.weights / weight_tanh_scale) + 1) / 2
            stability = pre_l_2d.swapaxes(-1,-2) * stability # only active inputs count for stability
            if len(stability.shape) == 2:
                stability = np.expand_dims(stability, 0) # handle single and multi inputs
            stability = np.sum(stability, axis = 0) / len(stability)
            #print(stability)
            self.stats = stability_mean * stability + (1-stability_mean) * self.stats
        return post_layer
        
    def change_weights(self, d_weights):
        if use_stability:
            direct = 1 - self.stats
        else:
            direct = 1
        #print('direct', direct)
        self.weights += d_weights * lr * direct
        np.clip(self.weights, -clip_weights, clip_weights, self.weights)
        
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
        
    def draw(self, result, usage = False):
        c = 0
        for layer in self.layers:
            c+=1
            print('layer',c)
            layer.draw(usage)
        if result is not None:
            if result[0] > 0:
                circle = pyplot.Circle((self.layers[-1].neurons[0].x, self.layers[-1].neurons[0].y), radius=neuron_radius+0.3, fill=False, color='green', linewidth = 3)
            else:
                circle = pyplot.Circle((self.layers[-1].neurons[0].x, self.layers[-1].neurons[0].y), radius=neuron_radius+0.3, fill=False, color='gray', linewidth = 3)
            pyplot.gca().add_patch(circle)
        pyplot.axis('scaled')
        pyplot.show()
        
    def predict(self, new_input, oo = None, drawit=False, usage = False):
        self.set_input(new_input, oo)
        prediction = self.forward(False)
        if oo is not None:
            self.error = oo - prediction
        if drawit:
            self.draw(oo, usage)
        return prediction
        
NN2 = DrawNet()
NN2.add_layer(3, np.random.rand(inputs.shape[1], hidden_size) - 0.5, None)
NN2.add_layer(hidden_size, np.random.rand(hidden_size, hidden_size), None)
NN2.add_layer(hidden_size, np.random.rand(hidden_size, 1)- 0.5, None)
NN2.add_layer(1, None, None)
NN2.set_input(inputs, outputs)

# train neural network
#NN2.train()


#testing single inputs for few shot learning
error_history = []
epoch_list = []
askuser = True
stopit = False
few_shot = False
for epoch in range(100):
    for i in range(len(inputs)):
        same = True
        first = True
        while same:
            if not few_shot:
                same = False
            if askuser:
                same = True
                NN2.predict(inputs[i], outputs[i], True, usage = False)
                # t = '3' 
                t = input(str(i)+' '+str(NN2.error)+' (1: same, 2:next, 3:stop asking, 4:exit, 5:few_shot)?')
                if t.isdigit():
                    t = int(t)
                    if t == 2:
                        same = False
                        break
                    if t == 3:
                        askuser = False
                        same = False
                    if t == 4:
                        stopit = True
                        break
                    if t == 5:
                        few_shot = True
                        askuser = False
            NN2.set_input(inputs[i:i+1], outputs[i:i+1])
            NN2.forward(dostats = first)
            NN2.backward()
            first = False
            error_history.append(sum(np.square(NN2.error)))
            epoch_list.append(epoch + i/8)
            if few_shot:
                if abs(NN2.error[0]) < few_shot_end:
                    break
        if stopit:
            break
    if stopit:
        break
    NN2.predict(inputs[0], outputs[0], True)
    

for i in range(len(inputs)):
    print(NN2.predict(inputs[i], outputs[i], drawit=True, usage = True), 'correct', outputs[i])

# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(epoch_list, error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

print('Error', error_history[-1])

