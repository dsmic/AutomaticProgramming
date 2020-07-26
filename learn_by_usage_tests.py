#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:45:54 2020

@author: detlef
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

based on: https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24
Created on Sun Jul 19 15:45:02 2020

@author: detlef
"""


import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
import random
from matplotlib import pyplot
from math import cos, sin, atan
#import numpy as np

def sigmoid(x):
    xx = x - 2
    return 1 / (1 + np.exp(-xx))

def sigmoid_derivative(x):
    xx = x - 2
    return (np.exp(-xx) / (np.exp(-xx) + 1) ** 2) #x * (1 - x) # this was an optimized derivative using the self.hidden



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
        self.values = values
        self.post_layer = None

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

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        if linewidth > 0:
            c = 'green'
        else:
            c = 'red'
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=np.sign(abs(linewidth)), color = c)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw(round(self.values[this_layer_neuron_index]))
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[previous_layer_neuron_index, this_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)
                    
    def backward(self, post_error):
        pre_error = np.dot(post_error * sigmoid_derivative(self.post_layer), self.weights.T)
        d_weights = np.dot(self.values.T, (2 * post_error * sigmoid_derivative(self.post_layer)))
        
        # this is tested from simple_nn
        # d_weights2 = np.dot(self.layer1.T, (2 * self.error * sigmoid_derivative(self.output)))
        # d_layer1 = np.dot(self.error * sigmoid_derivative(self.output), self.weights2.T)

        self.change_weights(d_weights)
        return pre_error # first idea to the layer backpropergation
    
    def forward(self, pre_layer):
        self.values = pre_layer
        if self.weights is None:
            return pre_layer
        post_layer = sigmoid(np.dot(pre_layer, self.weights))
        return post_layer
        
    def change_weights(self, d_weights):
        # d_weights must be handled here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pass

class DrawNet():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None, values=None):
        layer = Layer(self, number_of_neurons, weights, values)
        self.layers.append(layer)
    
    def forward(self):
        outp = self.layers[0].values
        for layer in self.layers:
            outp = layer.forward(outp)
        #self.layers[-1].values = outp
        return outp
    
    def backward(self, inp, outp):
        pre_error = self.layers[-1].pre_layer - outp
        for i in range(len(self.layer)-1, 0, -1):
            (pre_error, d_weights) = self.layers[i].backward(self.layers[i-1], pre_error)        
        return pre_error
    
    
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
        pyplot.show()
        

lr = 0.001
hidden_size = 4

lm_w = 8
m_lm_w = -5

np.seterr(under='ignore', over='ignore')

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
outputs = np.array([[0], [0], [1], [0], [1], [0], [0], [1]])


# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, x, y):
        self.inputs  = x
        self.y = y
        # initialize weights as .50 for simplicity
        self.weights1   = np.random.rand(self.inputs.shape[1], hidden_size) 
        self.weights2   = np.random.rand(hidden_size, 1)                 
        self.error_history = []
        self.epoch_list = []
        self.calls = None
        self.network = DrawNet()
        self.network.add_layer(3, self.weights1, None)
        self.network.add_layer(hidden_size, self.weights2, None)
        self.network.add_layer(1, None, None)
             
    # data will flow through the neural network.
    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.inputs, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

 
    # going backwards through the network to update weights
    def backpropagation(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        self.error = self.y - self.output
        d_weights2 = np.dot(self.layer1.T, (2*(self.error) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.inputs.T,  (np.dot(2*(self.error) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1 * lr
        self.weights2 += d_weights2 * lr
        
        self.weights1 = self.weights1.clip(m_lm_w, lm_w)
        self.weights2 = self.weights2.clip(m_lm_w, lm_w)
        
        #print(d_weights1)
        
    # train the neural net for 25,000 iterations
    def train(self, epochs=100000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.sum(np.square(self.error)))
            self.epoch_list.append(epoch)
    def init_stats(self):
        self.calls = 0
        self.stats1 = np.zeros(self.weights1.shape)
        self.stats2 = np.zeros(self.weights2.shape)
        
    def scale_stats(self, factor = 0.1):
        self.calls *= factor
        self.stats1 *= factor
        self.stats2 *= factor
    def random_weights(self):
        init_nonzero = 0.1
        for l in range(len(NN.weights1.flat)):
            if random.random() < init_nonzero:
                NN.weights1.flat[l] = random.choice([m_lm_w, lm_w])
            else:
                NN.weights1.flat[l] = 0
        for l in range(len(NN.weights2.flat)):
            if random.random() < init_nonzero:
                NN.weights2.flat[l] = random.choice([m_lm_w, lm_w])
            else:
                NN.weights2.flat[l] = 0
    def print_stats(self):
        for l in self.stats1.flat:
            print(l/self.calls, end=' ')
        print()
        for l in self.stats2.flat:
            print(l/self.calls, end=' ')
        print()
    
    def learn_test(self, limit = 0.05):
        init_nonzero = 0.2
        clear_nonzero = 0.01
        for l in range(len(NN.weights1.flat)):
            if NN.stats1.flat[l] < limit:
                if random.random() < init_nonzero:
                    NN.weights1.flat[l] = random.choice([m_lm_w, lm_w])
                else:
                    NN.weights1.flat[l] = 0
            else:
                if random.random() < clear_nonzero:
                    NN.weights1.flat[l] = 0
        
        for l in range(len(NN.weights2.flat)):    
            if NN.stats2.flat[l] < limit:
                if random.random() < init_nonzero:
                    NN.weights2.flat[l] = random.choice([m_lm_w, lm_w])
                else:
                    NN.weights2.flat[l] = 0
            else:
                if random.random() < clear_nonzero:
                    NN.weights2.flat[l] = 0
    
    # function to predict output on new and unseen input data                               
    def predict(self, new_input, oo = None, drawit=False):
        self.layer1 = sigmoid(np.dot(new_input, self.weights1))
        self.prediction = sigmoid(np.dot(self.layer1, self.weights2))
        if oo is not None:
            if self.calls is not None and len(new_input.shape) == 1:
                self.calls += 1
                for i in range(len(new_input)):
                    self.stats1[i] += np.sign(abs(self.weights1[i])) * np.sign(round(new_input[i])) * round(abs(oo[0]-self.prediction[0]))
                for i in range(len(self.layer1)):
                    self.stats2[i] += np.sign(abs(self.weights2[i])) * np.sign(round(self.layer1[i])) * round(abs(oo[0]-self.prediction[0]))
            if self.calls is not None and len(new_input.shape) == 2:
                for j in range(len(new_input)):
                    self.calls += 1
                    for i in range(len(new_input[j])):
                        self.stats1[i] += np.sign(abs(self.weights1[i])) * np.sign(round(new_input[j][i]))  * round(abs(oo[j][0]-self.prediction[j][0]))
                    for i in range(len(self.layer1[j])):
                        self.stats2[i] += np.sign(abs(self.weights2[i])) * np.sign(round(self.layer1[j][i]))  * round(abs(oo[j][0]-self.prediction[j][0]))
        if drawit:
                """
                network = DrawNet()
                # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)
                weights1 = self.weights1 #.T #np.array([\
                                     #[0,0,0],\
                                     #[0,0,0],\
                                     #[0,0,-8],\
                                     #[0,1,0]])
                weights2 = self.weights2 #.T #np.array([[1,1,1,1]])
            
            
                network.add_layer(3, weights1, new_input)
                network.add_layer(hidden_size, weights2, self.layer1)
                network.add_layer(1, None, self.prediction)
                """
                network = self.network
                network.layers[0].values = new_input
                print('checking',network.forward(), self.prediction)
                network.draw(oo)
        return self.prediction

# create neural network   
NN = NeuralNetwork(inputs, outputs)

"""
# train neural network
NN.train()

# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

print('Error',NN.error_history[-1])
"""

NN.init_stats()
#for i in range(len(inputs)):
#    print(NN.predict(inputs[i], outputs[i]), 'correct', outputs[i])
#print(NN.calls, NN.stats1, NN.stats2)

#print(NN.predict(inputs, outputs), 'correct', outputs)
#print(NN.calls, NN.stats1, NN.stats2)

NN.random_weights()        
print(NN.weights1)
print(NN.weights2)

test_min = 100
for _ in range(1000):
    rr = [0] * len(inputs)
    for i in range(len(inputs)):
        NN.learn_test()
        rr[i] = NN.predict(inputs[i], outputs[i], drawit=False)
        #inp = input('-')
    NN.scale_stats()
    err = sum((rr-outputs)**2)
    if err < test_min:
        print(rr, err, NN.weights1, NN.weights2)
        NN.print_stats()
        test_min = err

for i in range(len(inputs)):
    print(NN.predict(inputs[i], outputs[i], drawit=True), 'correct', outputs[i])



minimum = 100

more = True

next = {}
next[m_lm_w] = 0
next[0] = lm_w


w1_size = NN.weights1.size
w2_size = NN.weights2.size
all_size = w1_size + w2_size
print('all_size', all_size)

w = [m_lm_w] * all_size

while more:
    pos = 0
    while w[pos] == lm_w: # end reached
        w[pos] = m_lm_w
        pos += 1           
        if pos == all_size:
            break
    if pos == all_size:
        break    
    w[pos] = next[w[pos]]
    
    NN.weights1.flat[:] = w[:w1_size]
    NN.weights2.flat[:] = w[w1_size:]
    errsum = np.sum((NN.predict(inputs, outputs)-outputs)**2)
    if errsum < minimum:
        minimum = errsum
        for i in range(all_size):
            print(w[i],end=' ')
        
        print(errsum)
        for i in range(len(inputs)):
            print(NN.predict(inputs[i], drawit=True, oo=outputs[i]), 'correct', outputs[i])
        #print(np.sum((NN.predict(inputs)-outputs)**2))
        
        
