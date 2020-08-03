#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
based on: https://towardsdatascience.com/inroduction-to-neural-networks-in-python-7e0b422e6c24
    and https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network/29889993
    and https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
Created on Sun Jul 19 15:45:02 2020

@author: detlef




loss function used = 1/2 SUM(error**2) // making the derivative error
"""

import cupy as np # helps with the math (Faster for hidden_size > 256 probably)
#import numpy as np # helps with the math
from matplotlib import pyplot
from math import cos, sin, atan
import random

pyplot.rcParams['figure.dpi'] = 150
pyplot.interactive(False) # seems not to fix memory issue


verbose = 1

do_check_all = 0 #1000            # 0 to turn off

multi_test = -1 #1000             # 0 to turn off
max_iter = 30


hidden_size = 16
two_hidden_layers = True
use_bias = False

lr = 0.3
use_stability = False
stability_mean = 0.1
clip_weights = 1
clip_bias = 1
init_rand_ampl = 0.1
init_rand_ampl0 = 2# for first layer

scale_linewidth = 0.1
weight_tanh_scale = 0.1
scale_for_neuron_diff = 1

scale_sigmoid = 3
shift_sigmoid = 1

few_shot_end = 0.2
few_shot_max_try = 100



test_from_random_input = False
i_bits = 16

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
outputs = np.array([[0], [0], [1], [0], [1], [1], [0], [1]])

do_pm = False

load_mnist = True

do_batch_training = 20000

first_n_to_use = 60000
label_to_one = 5


num_outputs = 10 # most early test need this to be 1, later with mnist dataset this can be set to 10 eg.


#np.seterr(under='ignore', over='ignore')

def sigmoid(x):
    if do_pm:
        return np.tanh(x)
    xx = scale_sigmoid * (x - shift_sigmoid)
    return 1 / (1 + np.exp(-xx)) #* 2 -1

def sigmoid_derivative(x):
    if do_pm:
        return 1-np.tanh(x)**2
    xx = scale_sigmoid * (x - shift_sigmoid)
    return scale_sigmoid * (np.exp(-xx) / (np.exp(-xx) + 1) ** 2)

def transform_01_mp(x):
    return 2*x - 1

def run_load_mnist(show_msg = True, use_test = False):
    #global inputs, outputs, bbs
    # simelar to https://www.python-course.eu/neural_network_mnist.php
    import pickle
    #image_size = 28 # width and length
    #no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
    #image_pixels = image_size * image_size
    data_path = "/home/detlef/AutomaticProgramming/test_few_shot/data/mnist/" # makes it possible to use kernel from jupyter notebook
    
    # speedup loading
    try:
        with open(data_path + "pickled_mnist.pkl", "br") as fh:
            (train_data, test_data) = pickle.load(fh)
    except:
        train_data = np.loadtxt(data_path + "mnist_train.csv", 
                                delimiter=",")
        test_data = np.loadtxt(data_path + "mnist_test.csv", 
                           delimiter=",") 
        with open(data_path + "pickled_mnist.pkl", "bw") as fh:
            pickle.dump((train_data, test_data), fh)
            
    fac = 0.99 / 255

    if use_test:
        used_imgs = np.array(test_data[:, 1:]) * fac + 0.01
        dataset_name = 'Test dataset'
        used_labels = np.array(test_data[:, :1])
    else:
        used_imgs = np.array(train_data[:, 1:]) * fac + 0.01
        dataset_name = 'Train dataset'
        used_labels = np.array(train_data[:, :1])
    
    
    #train_labels = np.array(train_data[:, :1])
    
    
    # for i in range(10):
    #     print(train_labels[i])
    #     img = train_imgs[i].reshape((28,28))
    #     pyplot.imshow(img, cmap="Greys")
    #     pyplot.show()
        
    #train_labels = np.around(1 - np.sign(np.abs(train_labels - label_to_one)))        
    if num_outputs == 1:
        used_labels = np.around(1 - np.sign(np.abs(used_labels - label_to_one)))
    elif num_outputs == 10:
        label_transform = np.zeros((10, 10), int)
        np.fill_diagonal(label_transform, 1)
        label_transform = label_transform.tolist()
        used_labels = [label_transform[int(np.around(x[0]))] for x in used_labels]
        used_labels = np.array(used_labels)
#        raise Exception('not yet implementd')
    else:
        raise Exception('not yet implementd')

    inputs = used_imgs[:first_n_to_use]     
    outputs = used_labels[:first_n_to_use]
    bbs = ''
    for l in np.around(outputs):
        bbs += str(int(l[0]))
    if show_msg:
        if num_outputs == 10:
            print('loaded mnist', dataset_name,' with 10 labels')
        else:
            print('loaded mnist', dataset_name,' with output labels ',label_to_one,' resulting in learning labels', bbs[:10], '....')
    if len(bbs) > 50:
        bbs = 'to long to plot'
    if do_pm:
        inputs = transform_01_mp(inputs)
        outputs = transform_01_mp(outputs)
    return (inputs, outputs, bbs)

    
if load_mnist:
    (inputs, outputs, bbs) = run_load_mnist()
else:
    if do_pm: # prepare the fixed inputs, load_mnist does it in the function
        inputs = transform_01_mp(inputs)
        outputs = transform_01_mp(outputs)




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
    def __init__(self, network, number_of_neurons, weights, bias, values, slow_learning):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights
        if weights is not None:
            self.stability = np.zeros(weights.shape)
        self.bias = bias
        self.values = values
        self.slow_learning = slow_learning

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
                    stability = self.previous_layer.stability[previous_layer_neuron_index, this_layer_neuron_index]
                    used = 0
                    if weight > 0:
                        used = self.previous_layer.values[previous_layer_neuron_index] * self.values[this_layer_neuron_index]
                    else:
                        used = self.previous_layer.values[previous_layer_neuron_index] * (1 - self.values[this_layer_neuron_index])
                                                              
                    #print("connection %2d %2d    %6.3f    %6.3f    %6.3f    %6.3f used: %6.3f" % (previous_layer_neuron_index, this_layer_neuron_index, self.previous_layer.values[previous_layer_neuron_index], self.values[this_layer_neuron_index], weight, stability, used))
                    if usage:
                        self.__line_between_two_neurons(neuron, previous_layer_neuron, 4, usage = used)
                    else:
                        self.__line_between_two_neurons(neuron, previous_layer_neuron, 4, stability)
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)
                    
    def backward(self, post_error):
        
        error_between_sigmoid_and_full = post_error * sigmoid_derivative(self.between_full_sigmoid) # post layer may be wrong!!!!!!!!
        
        pre_error = np.dot(error_between_sigmoid_and_full, self.weights.T) 
        d_weights = np.dot(self.values.T, error_between_sigmoid_and_full) / len(post_error) # scale learning rate per input
        d_bias = np.sum(error_between_sigmoid_and_full, axis = 0) /len(post_error) 
        
        self.change_weights(d_weights, d_bias)
        return pre_error
    
    def forward(self, pre_layer, dostability):
        self.values = pre_layer
        if self.weights is None:
            return pre_layer
        self.between_full_sigmoid = np.dot(pre_layer, self.weights)
        if use_bias:
            self.between_full_sigmoid += self.bias
        post_layer = sigmoid(self.between_full_sigmoid)
        if dostability:
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
            #self.stability = stability_mean * pre_layer.T * stability + (1 - stability_mean * pre_layer.T) * self.stability
            self.stability = stability_mean * stability + (1 - stability_mean) * self.stability
        return post_layer
        
    def change_weights(self, d_weights, d_bias):
        if use_stability:
            direct = 1 - self.stability
        else:
            direct = np.array([1])
        #print('direct', direct)
        self.weights += d_weights * lr * direct * self.slow_learning
        self.bias +=  d_bias *lr * np.sum(direct, axis = 0) * self.slow_learning
        np.clip(self.weights, -clip_weights, clip_weights, self.weights)
        np.clip(self.bias, -clip_bias, clip_bias, self.bias)
        
class DrawNet():
    def __init__(self):
        self.layers = []
        self.epoch_list = []
        self.error_history = []
        self.error = None

        # batch handling, as cuda might not have enough memory to hold all inputs
        self.all_input = None
        self.all_output = None
        self.batch_pos = None
        self.batch_size = None
        
    def add_layer(self, number_of_neurons, weights, bias, values, slow_learning = 1):
        layer = Layer(self, number_of_neurons, weights, bias, values, slow_learning)
        self.layers.append(layer)
    
    def forward(self, dostability = False):
        outp = self.layers[0].values
        for layer in self.layers:
            outp = layer.forward(outp, dostability)
        #self.layers[-1].values = outp
        self.error = self.y - self.layers[-1].values
        return outp
    
    def backward(self):
        #self.error = pre_error = self.y - self.layers[-1].values # forward must be called first anyway
        pre_error = self.error
        for layer in reversed(self.layers[:-1]):
            #print('pre_error', pre_error.flatten())
            pre_error = layer.backward(pre_error)
        return pre_error
    
    def train(self, epochs=1000):
        self.epochs = epochs # just to know how it was trained for output
        for epoch in range(epochs):
            # flow forward and produce an output
            self.forward(True)
            # go back though the network to make corrections based on the output
            self.backward()
            self.next_batch()
            # keep track of the error history over each epoch
            self.error_history.append(np.sum(np.square(self.error)))
            self.epoch_list.append(epoch)
    
    def set_input(self, new_input, new_output, batch_size = None):
        self.all_input = new_input
        self.all_output = new_output
        self.batch_size = batch_size
        if batch_size is not None:
            self.layers[0].values = new_input[:batch_size]
            self.y = new_output[:batch_size]
        else:
            self.layers[0].values = new_input
            self.y = new_output
        self.batch_pos = self.batch_size
        
    def next_batch(self):
        if self.batch_size is not None:
            self.layers[0].values = self.all_input[self.batch_pos : self.batch_pos+self.batch_size]
            self.y = self.all_output[self.batch_pos : self.batch_pos+self.batch_size]
            # if len(self.y) == 0:
            #     self.batch_pos = self.batch_size
            #     self.layers[0].values = self.all_input[:self.batch_pos]
            #     self.y = self.all_output[:self.batch_pos]
            if len(self.y) < self.batch_size:
                self.batch_pos = self.batch_size - len(self.y)
                self.layers[0].values = np.concatenate((self.layers[0].values, self.all_input[:self.batch_pos]))
                self.y = np.concatenate((self.y, self.all_output[:self.batch_pos]))
            else:
                self.batch_pos += self.batch_size
            
            
    def draw(self, result, usage = False, display_title = None):
        c = 0
        for layer in self.layers:
            c+=1
            #print('layer',c)
            layer.draw(usage)
        if result is not None:
            if result[0] > 0:
                circle = pyplot.Circle((self.layers[-1].neurons[0].x, self.layers[-1].neurons[0].y), radius=neuron_radius+0.3, fill=False, color='green', linewidth = 3)
            else:
                circle = pyplot.Circle((self.layers[-1].neurons[0].x, self.layers[-1].neurons[0].y), radius=neuron_radius+0.3, fill=False, color='gray', linewidth = 3)
            pyplot.gca().add_patch(circle)
        pyplot.axis('scaled')
        if display_title is not None:
            pyplot.title(display_title)
        pyplot.show()
        pyplot.close()
        
    def predict(self, new_input, oo = None, drawit=False, usage = False, display_title = None):
        self.set_input(new_input, oo)
        prediction = self.forward(False)
        if oo is not None:
            self.error = oo - prediction
        if drawit:
            self.draw(oo, usage, display_title)
        return prediction
    
    def count_parameters(self):
        count = 0
        for l in self.layers:
            if l.weights is not None:
                count += l.weights.size
                if use_bias:
                    count += l.bias.size
        return count

def setup_net():
    NN2 = DrawNet()
    NN2.add_layer(len(inputs[0]), init_rand_ampl0 * (np.random.rand(inputs.shape[1], hidden_size) - 0.5), init_rand_ampl0 * (np.random.rand(hidden_size) - 0.5), None, slow_learning = 0.01)
    if two_hidden_layers:
        NN2.add_layer(hidden_size, init_rand_ampl * (np.random.rand(hidden_size, hidden_size) - 0.5), init_rand_ampl * (np.random.rand(hidden_size) - 0.5), None)
    NN2.add_layer(hidden_size, init_rand_ampl * (np.random.rand(hidden_size, num_outputs)- 0.5), init_rand_ampl * (np.random.rand(num_outputs) - 0.5), None)
    NN2.add_layer(num_outputs, None, None, None)
    NN2.set_input(inputs, outputs)
    print('Network parameters: ', NN2.count_parameters())
    return NN2


def creat_output_from_int(bb, length=8):
    output = [0]*length
    bbs = ('{0:0'+str(length)+'b}').format(bb)
    for l in range(len(bbs)): 
        if bbs[l] =='1':
            output[l] = [1]
        else:
            output[l] = [0]
    output = np.array(output)
    if do_pm:
        output = transform_01_mp(output)
    return output, bbs

if do_check_all > 0:
    notok = 0
    sum_error_history = None
    for bb in range(0, 256):
        # bbs = '{0:08b}'.format(bb)
        # for l in range(len(bbs)): 
        #     if bbs[l] =='1':
        #         outputs[l] = 1
        #     else:
        #         outputs[l] = 0
        (outputs, bbs) = creat_output_from_int(bb)
        NN2 = setup_net()
        NN2.train(do_check_all)
        err = np.sum(NN2.error**2)
        ok = '*'
        if err < 0.2: 
            ok = ' '
        else:
            notok += 1
        if sum_error_history is None:
            sum_error_history = np.array(NN2.error_history)
        else:
            sum_error_history += np.array(NN2.error_history)
        if verbose > 0:
            print(bbs, '{0:5.3f}'.format(float(err)),ok,notok)
            pyplot.figure(figsize=(15,5))
            pyplot.plot(NN2.epoch_list, NN2.error_history)
            pyplot.xlabel('Epoch')
            pyplot.ylabel('Error')
            pyplot.title(bbs)
            pyplot.show()
            pyplot.close()
    pyplot.figure(figsize=(15,5))
    pyplot.plot(NN2.epoch_list, (sum_error_history / 256).tolist())
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Error')
    pyplot.title('sum error history')
    pyplot.show()
    pyplot.close()
        
    import sys
    sys.exit()


        

# train neural network
#NN2.train()

#testing single inputs for few shot learning
askuser = True
stopit = False
few_shot = (multi_test > 0)


NN2 = setup_net()
multi = 0
sum_error_history = None
if do_batch_training > 0:
    NN2.set_input(inputs, outputs, batch_size=3000)
    NN2.train(do_batch_training)
    pyplot.figure(figsize=(15,5))
    pyplot.plot(NN2.epoch_list, (np.array(NN2.error_history) / len(outputs)).tolist())
    pyplot.xlabel('Batches')
    pyplot.ylabel('Error')
    pyplot.title('trained with epochs: ' + str(NN2.epochs))
    pyplot.show()
    pyplot.close()

    print('outputs', len(outputs), 'batch_size', NN2.batch_size, '1', int(np.sum(NN2.y > 0.5)), 'wrong', int(np.sum((NN2.y > 0.5) * (NN2.error**2 > 0.25))), 'Ratio', int(np.sum((NN2.y > 0.5) * (NN2.error**2 > 0.25))) / int(np.sum(NN2.y > 0.5)), 'Error', float(np.sum(NN2.error**2) / len(NN2.error)))
    (inputs, outputs, bbs) = run_load_mnist(use_test = True)
    NN2.set_input(inputs, outputs, batch_size=1000)
    NN2.forward()
    print('outputs', len(outputs), 'batch_size', NN2.batch_size, '1', int(np.sum(NN2.y > 0.5)), 'wrong', int(np.sum((NN2.y > 0.5) * (NN2.error**2 > 0.25))), 'Ratio', int(np.sum((NN2.y > 0.5) * (NN2.error**2 > 0.25))) / int(np.sum(NN2.y > 0.5)), 'Error', float(np.sum(NN2.error**2) / len(NN2.error)))
    

while multi <= multi_test:
    pos_under_few_shot = 0
    if test_from_random_input:
        inp = []
        while len(inp) < 8:
            r = random.randrange(0,2**i_bits-1)
            if r not in inp:
                inp.append(r)
        inputs = []
        for bb in inp:
            v= [0]*i_bits
            bbs = ('{0:0'+str(i_bits)+'b}').format(bb)
            for l in range(len(bbs)): 
                if bbs[l] =='1':
                    v[l] = 1
                else:
                    v[l] = 0
            inputs.append(v)
        inputs = np.array(inputs)
    if not load_mnist:
        (outputs, bbs) = creat_output_from_int(random.randrange(0,255))
    
    # reset weights in the last layer
    NN2.layers[-2].weights = init_rand_ampl * (np.random.rand(hidden_size, 1)- 0.5)
    NN2.layers[-2].bias = init_rand_ampl * (np.random.rand(1)- 0.5)
    label_to_one = random.randrange(0, 9) # change the label used
    run_load_mnist(False)
    NN2.set_input(inputs, outputs)
    # used to reset whole NN every time
    #NN2 = setup_net()

    error_history = []
    epoch_list = []
    epoch = 0
    while epoch < max_iter:
        for i in range(len(inputs)):
            same = True
            first = True
            fl = 0
            while same:
                if not few_shot:
                    same = False
                if askuser and multi_test == 0:
                    same = True
                    NN2.predict(inputs[i], outputs[i], True, usage = False)
                    # t = '3' 
                    doask = True
                    while doask:
                        doask = False
                        t = input(str(i)+' '+str(NN2.error)+' (1: same, 2:next, 3:stop asking, 4:exit, 5:few_shot, 6: change max epoch num)?')
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
                            if t == 6:
                                max_iter = int(input('change max epoch num ' + str(max_iter) + ' '))
                                doask = True
                NN2.set_input(inputs[i:i+1], outputs[i:i+1])
                NN2.forward(dostability = first)
                NN2.backward()
                first = False
                this_error = sum(np.square(NN2.error))
                if this_error[0] > few_shot_end:
                    pos_under_few_shot = epoch + 1
                if len(epoch_list) == 0 or (len(epoch_list) > 0 and epoch_list[-1] != epoch + i / len(inputs)):
                    epoch_list.append(epoch + i / len(inputs))
                    error_history.append(this_error)
                fl += 1
                if fl > few_shot_max_try:
                    break
                if few_shot:
                    if abs(NN2.error[0]) < few_shot_end:
                        break
            if stopit:
                break
        if stopit:
            break
        NN2.set_input(inputs, outputs)
        NN2.forward()
        err = outputs - NN2.layers[-1].values
        NN2.predict(inputs[0], outputs[0], multi_test == 0, display_title = str(epoch)+': '+'{0:6.3f}'.format(float(np.sum(err**2))))
        epoch += 1
    

    if multi_test == 0:
        for i in range(len(inputs)):
            print(NN2.predict(inputs[i], outputs[i], drawit= (multi_test > 0), usage = True), 'correct', outputs[i])
    
    # plot the error over the entire training duration
    if sum_error_history is None:
        sum_error_history = np.array(error_history)
    else:
        sum_error_history += np.array(error_history)
    if verbose > 0:
        pyplot.figure(figsize=(15,5))
        pyplot.plot(epoch_list, error_history)
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Error')
        pyplot.title(str(multi)+ ' ' + bbs)
        pyplot.show()
        pyplot.close()
    
    print(multi, 'Label', label_to_one, 'Error', np.sum(np.array(error_history[-8:])), pos_under_few_shot)
    multi += 1
if sum_error_history is not None:
        pyplot.figure(figsize=(15,5))
        pyplot.plot(epoch_list, (sum_error_history / multi_test).tolist())
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Error')
        pyplot.title('sum error history')
        pyplot.show()
        pyplot.close()
