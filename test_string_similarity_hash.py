#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:22:15 2020

@author: detlef
"""

#import numpy as np
import tokenize
from matplotlib import pyplot
from tqdm import tqdm

use_cuda = False
if use_cuda:
    import cupy as np # helps with the math (Cuda supported: faster for hidden_size > 256 probably and most mnist cases with batch training)
    import cupyx.scipy.sparse as sparse
else:
    import numpy as np # helps with the math (if no Cuda is availible or size is small for simple tests)
    import scipy.sparse as sparse


hidden_size = 100
lr =            0.0002
clip_weights = 1 # (clipping to 1 was used for most tests)
clip_bias = None # 1
two_hidden_layers = True


few_shot_more_at_once = 5
few_shot_max_try = 1000
few_shot_threshold_ratio = 1.2 # for mnist
few_shot_threshold = 0.2


verbose = 0


our_dtype = np.float32   # float32 is 30 times faster on batch training with GTX1070Ti and 3 times faster than i7-4790K with float64, cpu does not help float32 a lot, this varys a lot with sizes and if stability is used.)
def np_array(x):
    return np.array(x, dtype = our_dtype) 
check_for_nan = True


class net_database():
    def __init__(self):
        self.data = {}
        self.keys = np.empty([0,64])
        
    
    def string_to_simelar_has(self, x):
        def string_to_bin_narray(x):
            xh = hash(x)
            if xh > 0:
                b = list(bin(hash(x)))[2:]
                b = [0] * (64-len(b)) + b
            else:
                b = list(bin(hash(x)))[3:]
                b = [1] + [0] * (63-len(b)) + b
            b = np.array(b, dtype = float) - 0.5
            return b
        
        h = np.array([0] * 64, dtype = float)
        if len(x) <= 3:
            return string_to_bin_narray(x)
        for i in range(len(x)-3):
            h += string_to_bin_narray(x[i:i+3])
        return np.tanh(h)/2+0.5

    def add_data(self, x):
        key = self.string_to_simelar_has(x)
        key_tuple = tuple(key)
        #print(x,key)
        if key_tuple not in self.data:
            self.data[key_tuple] = x
            self.keys = np.append(self.keys, key.reshape((1,64)), axis = 0)
        return key
    
    def get_data_key(self, key):
        dist = np.sum((self.keys - key)**2, axis = 1)
        m = dist.argmin()
        print(dist, m)
        return self.data[tuple(self.keys[m])]
        
    def get_data_string(self, x):
        key = self.string_to_simelar_has(x)
        return self.get_data_key(key)
    
np.set_printoptions(precision=2, suppress = True, linewidth=150)
    

store = net_database()

store.add_data("hallo")
store.add_data("wer ist")
store.add_data("ist noch was")

a = store.get_data_string("ist noch was")
print(a)


def read_file(file_name):
    txt_file = open(file_name)
    return [line.strip('\n') for line in txt_file]

limit_files = 100

def load_dataset(file_names, save_tokens_file_num = 0):
    data_set = []
    count = 0
    for file_name in file_names:
        if limit_files > 0 and count > limit_files:
            break
        try:
            python_file = open(file_name)
            py_program = tokenize.generate_tokens(python_file.readline) # just check if no errors accure
            d = list(py_program)
            # data_set += d
            for t in d:
                key = store.add_data(t.string)
                data_set.append(key.tolist())
            count += 1
        except UnicodeDecodeError as e:
            print(file_name + '\n  wrong encoding ' + str(e))
        except tokenize.TokenError as e:
            print(file_name + '\n  token error ' + str(e))
    return data_set


train_data_set = load_dataset(read_file('python100k_train.txt'))    
test_data_set = load_dataset(read_file('python50k_eval.txt'))    


use_stability = False
from math import cos, sin, atan
# drawing parameters
scale_linewidth = 0.1
weight_tanh_scale = 0.1
scale_for_neuron_diff = 1
scale_sigmoid = 3
shift_sigmoid = 1
sparse_layers = []
scale_lr_treshold = 0.05

def dot_sparse_result(Mask,U,V):
    if isinstance(Mask, np.ndarray):
        return np.dot(U,V)
    
    # use Mask to select elements of U and V
    A3=sparse.coo_matrix(Mask, dtype=our_dtype) 
    
    # slower
    #for i in range(len(A3.data)):
    #    A3.data[i] = np.inner(U[A3.row[i],:],V[:,A3.col[i]])
    #return A3
    
    # U1=U[A3.row,:]
    # V1=V[:,A3.col]
    # A3.data[:] = np.einsum('ij,ji->i', U1, V1, optimize='optimal')
    # return A3
    
    for i in range(0, len(A3.data), 1000):
        U1=U[A3.row[i:i+1000],:]
        V1=V[:,A3.col[i:i+1000]]
        A3.data[i:i+1000] = np.einsum('ij,ji->i', U1, V1, optimize='optimal')
    return A3

use_bias = False
do_pm = False
def sigmoid(x):
    if do_pm:
        return np.tanh(x)
    xx = scale_sigmoid * (x - shift_sigmoid)
    #np.clip(xx,-10,10,xx) # avoid under over flow
    return 1 / (1 + np.exp(-xx)) #* 2 -1

def sigmoid_derivative(x):
    if do_pm:
        return 1-np.tanh(x)**2
    xx = np.exp(-scale_sigmoid * (x - shift_sigmoid))
    return scale_sigmoid * (xx / (xx + 1) ** 2)

def transform_01_mp(x):
    return 2*x - 1

stability_mean = 0.1
disable_progressbar = False

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
            if use_stability:
                self.stability = np.zeros(weights.shape)
            self.mask = weights.copy()
        self.drop_weights = None
        self.bias = bias
        self.values = values
        self.slow_learning = slow_learning
        
        self.d_weights = None
        self.d_weights_ratio = None

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
            neuron.draw(np.around(self.values[this_layer_neuron_index]))
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
                        if use_stability:
                            self.__line_between_two_neurons(neuron, previous_layer_neuron, 4, stability)
                        else:
                            self.__line_between_two_neurons(neuron, previous_layer_neuron, 4, 0.3)
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)
                    
    def backward(self, post_error, direction_factor = 1):
        
        #error_between_sigmoid_and_full = post_error * sigmoid_derivative(self.between_full_sigmoid) # this is the straight forward way of the derivative
        
        error_between_sigmoid_and_full = post_error * scale_sigmoid * self.post_layer * (1 - self.post_layer) # this version of the derivative uses the result from forward
        
        if len(sparse_layers) == 0:
            pre_error = np.dot(error_between_sigmoid_and_full, self.weights.T) 
        else:
            pre_error = self.weights.dot(error_between_sigmoid_and_full.T).T
        
        if self.slow_learning != 0: # speed up, if learning is disabled
            if scale_lr_treshold > 0 and self.d_weights is not None:
                last_d_weights = self.d_weights
            else:
                last_d_weights = None
            if len(sparse_layers) == 0:
                self.d_weights = np.dot(self.values.T, error_between_sigmoid_and_full) 
                # d_weights *= 1 / len(post_error) # scale learning rate per input
            else:
                self.d_weights = dot_sparse_result(self.mask, self.values.T, error_between_sigmoid_and_full) 
                # d_weights *=   1 / len(post_error)  # scale learning rate per input
            if last_d_weights is not None:
                mean_len_d_weights = (np.linalg.norm(last_d_weights) + np.linalg.norm(self.d_weights)) / 2
                diff_d_weights = np.linalg.norm(last_d_weights - self.d_weights)
                self.d_weights_ratio = diff_d_weights / mean_len_d_weights
                #print(self.d_weights_ratio)
            
            if use_bias:
                self.d_bias = np.sum(error_between_sigmoid_and_full, axis = 0) /len(post_error) 
        
            self.change_weights(direction_factor)
        return pre_error
    
    def forward(self, pre_layer, dostability):
        self.values = pre_layer
        if self.weights is None:
            return pre_layer
        if len(sparse_layers) == 0:
            self.between_full_sigmoid = np.dot(pre_layer, self.weights)
        else:
            self.between_full_sigmoid = self.weights.T.dot(pre_layer.T).T
        if use_bias:
            self.between_full_sigmoid += self.bias
        self.post_layer = sigmoid(self.between_full_sigmoid)
        if dostability:
            post_l = np.expand_dims(self.post_layer,-2)
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
        return self.post_layer
        
    def change_weights(self, direction_factor):
        if use_stability:
            direct = 1 - self.stability
            self.weights += self.d_weights *( lr * direct * self.slow_learning * direction_factor)
            self.bias +=  self.d_bias * np.sum(direct, axis = 0) * (lr * self.slow_learning  * direction_factor)
        else:
            self.weights += self.d_weights * (lr * self.slow_learning  * direction_factor)
            if use_bias:
                self.bias +=  self.d_bias * (lr * self.slow_learning  * direction_factor)
        if clip_weights is not None:
            if isinstance(self.weights, np.ndarray):
                np.clip(self.weights, -clip_weights, clip_weights, self.weights)
            else:
                np.clip(self.weights.data, -clip_weights, clip_weights, self.weights.data)
        if clip_bias is not None:
            np.clip(self.bias, -clip_bias, clip_bias, self.bias)
        if self.drop_weights is not None:
            self.weights *= self.drop_weights
            
        
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
        if check_for_nan:
            if np.any(np.isnan(self.error)):
                print('nan')
        return outp
    
    def backward(self, direction_factor = 1):
        #self.error = pre_error = self.y - self.layers[-1].values # forward must be called first anyway
        pre_error = self.error
        for layer in reversed(self.layers[:-1]):
            #print('pre_error', pre_error.flatten())
            pre_error = layer.backward(pre_error, direction_factor)
        return pre_error
    
    def change_weights(self,grad_direct):
        for layer in self.layers[:-1]:
            layer.change_weights(grad_direct)
    
    def loss(self):
        return np.sum(np.square(self.error))
    
    def train(self, epochs=1000):
        self.epochs = epochs # just to know how it was trained for output
        self.error_history = []
        self.epoch_list = []
        ttt = tqdm(range(epochs), mininterval = 10, disable=disable_progressbar)
        for epoch in ttt:
            # flow forward and produce an output
            self.forward(use_stability)
            # go back though the network to make corrections based on the output
            self.backward()
            self.next_batch()
            # keep track of the error history over each epoch
            err = self.loss()
            self.error_history.append(err)
            self.epoch_list.append(epoch)
            if self.batch_size is not None:
                ttt.set_description("Err %6.3f" % (err/self.batch_size), refresh=False)
            else:
                ttt.set_description("Err %6.3f" % (err), refresh=False)
        self.forward() # to update the output layer, if one needs to print infos...
    
    def train_2(self, shots=1000):
        self.epochs = shots # just to know how it was trained for output
        self.error_history = []
        self.epoch_list = []
        ttt = tqdm(range(shots), mininterval = 10, disable=disable_progressbar)
        for shot in ttt:
            # flow forward and produce an output
            begin = shot - few_shot_more_at_once
            if begin < 0:
                begin = 0
            self.layers[0].values = self.all_input[begin:shot+1]
            self.y = self.all_output[begin:shot+1]
            
            self.one_shot_3()
            
            err = self.loss()
            self.error_history.append(err)
            self.epoch_list.append(shot)
            ttt.set_description("Err %6.3f" % (err), refresh=False)
            
        self.forward() # to update the output layer, if one needs to print infos...
    
    
    def one_shot(self):
        epoch = 0
        while epoch < few_shot_max_try:
            self.forward()
            # criterium for stopping is only used for the first element, which is the one few shot is done for. The other elements are not checked, but only used for stabilizing old learned data
            if (self.layers[-1].values.argmax(axis = 1) == self.y.argmax(axis=1))[0]:
                biggest_two = np.partition(self.layers[-1].values[0], -2)[-2:]
                if do_pm:
                    ratio = (biggest_two[-1] + 1) / (biggest_two[-2] + 1) / 2 # do_pm means rsults between -1 and 1
                else:
                    ratio = biggest_two[-1] / biggest_two[-2]
                if verbose > 0:
                    print(biggest_two, ratio)
                if ratio > few_shot_threshold_ratio and biggest_two[-1] > few_shot_threshold:
                    break
            self.backward()
            epoch += 1
            
        return epoch < few_shot_max_try
    
    def one_shot_3(self):
        epoch = 0
        scale_lr = 1
        while epoch < few_shot_max_try:
            self.forward()
            # criterium for stopping is only used for the first element, which is the one few shot is done for. The other elements are not checked, but only used for stabilizing old learned data
            if (self.layers[-1].values.argmax(axis = 1) == self.y.argmax(axis=1))[0]:
                biggest_two = np.partition(self.layers[-1].values[0], -2)[-2:]
                if do_pm:
                    ratio = (biggest_two[-1] + 1) / (biggest_two[-2] + 1) / 2 # do_pm means rsults between -1 and 1
                else:
                    ratio = biggest_two[-1] / biggest_two[-2]
                if verbose > 0:
                    print(biggest_two, ratio)
                if ratio > few_shot_threshold_ratio and biggest_two[-1] > few_shot_threshold:
                    break
            self.backward(scale_lr)
            max_weight_ratio = None
            for l in self.layers:
                wr = l.d_weights_ratio
                if wr is not None:
                    if max_weight_ratio is not None:
                        if wr > max_weight_ratio:
                            max_weight_ratio = wr
                    else:
                        max_weight_ratio = wr
            if verbose > 0:
                print(max_weight_ratio, scale_lr)
            if max_weight_ratio is not None:
                if max_weight_ratio > scale_lr_treshold * 2:
                    scale_lr /= 2
                if max_weight_ratio < scale_lr_treshold:
                    scale_lr *= 2
                        
                
            epoch += 1
            
        return epoch < few_shot_max_try
    
    def one_shot_2(self):
        epoch = 0
        while epoch < few_shot_max_try:
            self.forward()
            old_loss = self.loss()
            self.backward()
            grad_direct = 1
            stop_it = False
            for devided in range(10):
                self.forward()
                new_loss = self.loss()
                while new_loss < old_loss:
                    old_loss = new_loss
                    self.change_weights(grad_direct)
                    self.forward()
                    new_loss = self.loss()
                    if verbose > 0:
                        print('newloss', old_loss, new_loss, new_loss-old_loss)
                    epoch += 1
                    if epoch >= few_shot_max_try:
                        stop_it = True
                        break
                    if (self.layers[-1].values.argmax(axis = 1) == self.y.argmax(axis=1))[0]:
                        biggest_two = np.partition(self.layers[-1].values[0], -2)[-2:]
                        if do_pm:
                            ratio = (biggest_two[-1] + 1) / (biggest_two[-2] + 1) / 2 # do_pm means rsults between -1 and 1
                        else:
                            ratio = biggest_two[-1] / biggest_two[-2]
                        if verbose > 0:
                            if verbose > 0:
                                print(biggest_two, ratio)
                        if ratio > few_shot_threshold_ratio and biggest_two[-1] > few_shot_threshold:
                            stop_it = True
                            break
                    
                if stop_it:
                    break
                old_loss = new_loss
                grad_direct *= -0.5
                self.change_weights(grad_direct)
                if verbose > 0:
                    print('new_direct', devided, grad_direct, old_loss)
            if stop_it:
                break
        if verbose > 0:
            print('ready')
                    
    
    
    def plot_train_history(self):
        pyplot.figure(figsize=(15,5))
        pyplot.plot(self.epoch_list, self.error_history)
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Error')
        pyplot.show()
        pyplot.close()

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
            
            
    def draw(self, result = None, usage = False, display_title = None):
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
        pyplot.xticks([])
        pyplot.yticks([])
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
        drops = 0
        for l in self.layers:
            if l.weights is not None:
                if isinstance(l.weights, np.ndarray):
                    count += l.weights.size
                else:
                    count += l.weights.data.size
                if use_bias:
                    count += l.bias.size
                if l.drop_weights is not None:
                    drops += l.drop_weights.size - np.sum(l.drop_weights)
        return count, drops


init_rand_ampl = 0.5
init_rand_ampl0 = [] # [1,0.5,1] #2 # for first layers    ([2] was used for most tests to make the first layer a mostly random layer)

inputs = np_array(train_data_set)[:-1,:]
outputs = np_array(train_data_set)[1:,:]
num_outputs = outputs[0].shape[0]
do_drop_weights = []

def setup_net():
    randamp = init_rand_ampl0.copy()
    randamp = randamp + [init_rand_ampl] * 5
    NN2 = DrawNet()
    input_len = inputs[0].shape[0]
    NN2.add_layer(input_len, randamp[0] * np_array(np.random.rand(input_len,hidden_size)), randamp[0] * np_array(np.random.rand(hidden_size) - 0.5), None)
    randamp.pop(0)
    if two_hidden_layers:
        NN2.add_layer(hidden_size, randamp[0] * np_array(np.random.rand(hidden_size, hidden_size)), randamp[0] * np_array(np.random.rand(hidden_size) - 0.5), None)
        randamp.pop(0)
    NN2.add_layer(hidden_size, randamp[0] * np_array(np.random.rand(hidden_size, num_outputs)), randamp[0] * np_array(np.random.rand(num_outputs) - 0.5), None)
    NN2.add_layer(num_outputs, None, None, None)
    NN2.set_input(inputs, outputs)
    count_drops = 0
    for l in range(len(do_drop_weights)):
        if do_drop_weights[l] > 0:
            NN2.layers[l].drop_weights = np.random.rand(NN2.layers[l].weights.size).reshape(NN2.layers[l].weights.shape) > do_drop_weights[l]
            count_drops += NN2.layers[l].drop_weights.size - np.sum(NN2.layers[l].drop_weights)
    num_params, count_drops = NN2.count_parameters()
    if verbose > 0:
        print('Network parameters: ', num_params, 'dropped', count_drops, 'real parameters', num_params - count_drops, 'drop definition', do_drop_weights)
    
    return NN2

setup_net()