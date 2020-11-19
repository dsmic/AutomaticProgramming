#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:04:58 2019

@author: detlef
"""
import os
import pickle
import argparse
#import tensorflow as tf
#from tensorflow.keras import Model
#from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
#from tensorflow.keras.layers import Activation, Embedding, Dense,  Lambda, Input
#from tensorflow.keras.layers import SimpleRNN, GRU, LSTM
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.optimizers import SGD
import numpy as np
import tokenize
#from collections import OrderedDict
from threading import Lock

import torch
import torch.nn as nn


#from tcn import TCN #, tcn_full_summary


#from sortedcontainers import SortedList

#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # remove some optimization warnings for tensorflow

parser = argparse.ArgumentParser(description='train recurrent net.')
parser.add_argument('--lr', dest='lr',  type=float, default=1e-2)
parser.add_argument('--epochs', dest='epochs',  type=int, default=500)
parser.add_argument('--hidden_size', dest='hidden_size',  type=int, default=50)
parser.add_argument('--final_name', dest='final_name',  type=str, default='final_model')
parser.add_argument('--pretrained_name', dest='pretrained_name',  type=str, default=None)
parser.add_argument('--attention', dest='attention', action='store_true')
parser.add_argument('--depth', dest='depth',  type=int, default=3)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--only_one', dest='only_one', action='store_true')
parser.add_argument('--revert', dest='revert', action='store_true')
parser.add_argument('--add_history', dest='add_history', action='store_true')
parser.add_argument('--RNN_type', dest='RNN_type',  type=str, default='GRU')
parser.add_argument('--gpu_mem', dest='gpu_mem',  type=float, default=1)
parser.add_argument('--fill_vars_with_atoms', dest='fill_vars_with_atoms', action='store_true')
parser.add_argument('--rand_files', dest='rand_files', action='store_true')
parser.add_argument('--float_type', dest='float_type',  type=str, default='float32')
parser.add_argument('--load_weights_name', dest='load_weights_name',  type=str, default=None)


parser.add_argument('--limit_files', dest='limit_files',  type=int, default=0)
parser.add_argument('--epoch_size', dest='epoch_size',  type=int, default=1000)
parser.add_argument('--max_length', dest='max_length',  type=int, default=20000)
parser.add_argument('--tensorboard_logdir', dest='tensorboard_logdir',  type=str, default='./logs')
parser.add_argument('--EarlyStop', dest='EarlyStop',  type=str, default='EarlyStop')
parser.add_argument('--embeddings_trainable', dest='embeddings_trainable', action='store_true')
parser.add_argument('--embed_len', dest='embed_len',  type=int, default=None)
parser.add_argument('--two_LSTM', dest='two_LSTM', action='store_true')
parser.add_argument('--token_number', dest='token_number',  type=int, default=1000)
parser.add_argument('--only_token_type', dest='only_token_type', action='store_true')
parser.add_argument('--remove_comments', dest='remove_comments', action='store_true')
parser.add_argument('--only_token_detail', dest='only_token_detail', action='store_true')
parser.add_argument('--only_token_detail_name', dest='only_token_detail_name', action='store_true')
parser.add_argument('--no_string_details', dest='no_string_details', action='store_true')
parser.add_argument('--enable_pretokenized_files', dest='enable_pretokenized_files', action='store_true')

parser.add_argument('--benchmark_parsing', dest='benchmark_parsing',   type=int, default=0)

parser.add_argument('--save_tokens_file_num', dest='save_tokens_file_num',  type=int, default=1000) #default, as it is much better than the other tests at the moment
parser.add_argument('--save_tokens_min_num', dest='save_tokens_min_num',  type=int, default=100)

parser.add_argument('--string_hash', dest='string_hash', action='store_true')
parser.add_argument('--keras_tcn', dest='keras_tcn', action='store_true')
parser.add_argument('--only_second_keras_tcn', dest='only_second_keras_tcn', action='store_true')
parser.add_argument('--only_first_keras_tcn', dest='only_first_keras_tcn', action='store_true')
args = parser.parse_args()


max_output = args.token_number

#RNN_type = {}
#RNN_type['LSTM'] = LSTM
#RNN_type['GRU'] = GRU
#RNN_type['SimpleRNN'] = SimpleRNN

#LSTM_use = RNN_type[args.RNN_type]

#tensorflow 2.0b sets memory growth per default, seems to be changed ?!
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)


class token_sort:
    def __init__(self, used_dict):
        self.np_sorted=[]
        self.used_dict = used_dict
        
    def add(self, entry):
        value = self.used_dict[entry]
        up_bound = len(self.np_sorted)
        low_bound = 0
        while (up_bound - low_bound > 1):
            pos = int((up_bound+low_bound) / 2)
            if self.used_dict[self.np_sorted[pos]] < value:
                up_bound = pos
            else:
                low_bound = pos
        if up_bound-low_bound > 0 and self.used_dict[self.np_sorted[low_bound]] < value:
            up_bound = low_bound
        self.np_sorted.insert(up_bound,entry)
        
    def delete(self, entry):
        value = self.used_dict[entry]
        up_bound = len(self.np_sorted)
        low_bound = 0
        while (up_bound - low_bound > 1):
            pos = int((up_bound+low_bound) / 2)
            if self.used_dict[self.np_sorted[pos]] < value:
                up_bound = pos
            else:
                low_bound = pos
        if up_bound-low_bound > 0 and self.used_dict[self.np_sorted[low_bound]] < value:
            up_bound = low_bound
        up_bound -= 1
        while (self.used_dict[self.np_sorted[up_bound]] == value):
            if self.np_sorted[up_bound] == entry:
                #print("remove", entry)
                self.np_sorted.pop(up_bound)
                return
            up_bound -= 1
        raise NameError('should not happen ',entry,'not found')
        
    def pop_lowest(self):
        return self.np_sorted.pop()
    
    def display(self):
        print('***************************')
        for i in self.np_sorted:
            print(i, self.used_dict[i])
        print('---------------------------')
            
    def check_order(self):
        p = 0
        for i in self.np_sorted:
            if self.used_dict[i] < p:
                self.display()
                return False
            p= self.used_dict[i]
        return True
            
        
class Token_translate:
    def __init__(self, num_free):
        self.data = {}
        self.used = {} #OrderedDict([])
        self.used_sorted = token_sort(self.used)
        self.back = {}
        self.free_numbers = [i for i in range(num_free)] 
        self.lock = Lock()
        self.found = 0
        self.calls = 0
        self.removed_numbers = {}
        self.num_free = num_free
        
    def translate(self,token):
        # seems to be called by different threads?!
        if args.remove_comments and (token[0] == tokenize.COMMENT or token[0] == tokenize.NL):
            #print('comment removed')
            return None
        s = token.string
        sh = hash(s)
        return sh % max_output # seems to be changed after initialization

    def get_string(self, num_of_token):
        return self.back[num_of_token]
            
translator = Token_translate(max_output)            


def read_file(file_name):
    txt_file = open(file_name)
    return [line.strip('\n') for line in txt_file]

all_tokens = {}
def load_dataset(file_names, save_tokens_file_num = 0):
    global num_ord
    data_set = []
    count = 0
    # py_programs = []    
    for file_name in file_names:
        count += 1
        if args.limit_files >0 and count > args.limit_files:
            break
        try:
            python_file = open(file_name)
            py_program = tokenize.generate_tokens(python_file.readline) # just check if no errors accure
            #list(py_program) #force tokenizing to check for errors
            #program_lines = []
            if save_tokens_file_num != 0:
                if count < save_tokens_file_num:
                    for py_token in py_program:
                        if py_token[0] not in all_tokens:
                            all_tokens[py_token[0]] = {}
                        in_tokens = all_tokens[py_token[0]]
                        if py_token[1] not in in_tokens:
                            in_tokens[py_token[1]] = 0
                        in_tokens[py_token[1]] += 1
                        #print(py_token)
                        #token_number = translator.translate(py_token)
                        #print("---",token_number, '- ' + translator.get_string(token_number)+' -')
                else:
                    list(py_program) #force tokenizing to check for errors
            else:
                list(py_program) #force tokenizing to check for errors
            data_set.append(file_name)
        except UnicodeDecodeError as e:
            print(file_name + '\n  wrong encoding ' + str(e))
        except tokenize.TokenError as e:
            print(file_name + '\n  token error ' + str(e))
#    return py_programs            
    return data_set

if args.rand_files:
    file_append="_random"
else:
    file_append=""
    
saved_tokens = {}
saved_tokens_howoften = {}
saved_tokens_howoften_id = {}
saved_pos = 0    
if args.save_tokens_file_num != 0:
    load_dataset(read_file('python100k_train.txt_random'), args.save_tokens_file_num)
    print('used for counting',args.save_tokens_file_num, 'min num', args.save_tokens_min_num)
    for i in all_tokens:
        t = all_tokens[i]
        c = 0
        saved_tokens[(i)] = saved_pos
        saved_pos += 1
        s = 0
        for l in t:
            s += t[l]
            if t[l] > args.save_tokens_min_num:
                saved_tokens[(i,l)] = saved_pos
                saved_tokens_howoften[(i, l)] = t[l] 
                saved_pos += 1
                c +=1
        saved_tokens_howoften_id[(i)] = s
        print(i, len(all_tokens[i]), c, len(saved_tokens))
    max_output = saved_pos + 1
    
summe = sum(saved_tokens_howoften_id.values())
print('numer of tokens for dictionary', summe)
for l in sorted(saved_tokens_howoften.items(), key=lambda x: x[1], reverse=False):
    print(l[0], l[1] / summe)

for l in sorted(saved_tokens_howoften_id.items(), key=lambda x: x[1], reverse=False):
    print(l[0], l[1] / summe)

for l in saved_tokens:
    print(l, saved_tokens[l])

if args.limit_files == 0 and os.path.exists('train.ddddd'):
    print('use preloaded train list !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    with open('train.ddddd','rb') as fb:
        train_data_set = pickle.load(fb)
else:    
    train_data_set = load_dataset(read_file('python100k_train.txt'+file_append))    
    if args.limit_files == 0:
        with open('train.ddddd','wb') as fb:
            pickle.dump(train_data_set, fb)
    
print(len(train_data_set))

if args.limit_files == 0 and os.path.exists('test.ddddd'):
    print('use preloaded test list !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    with open('test.ddddd','rb') as fb:
        test_data_set = pickle.load(fb)
else:    
    test_data_set = load_dataset(read_file('python50k_eval.txt'+file_append))    
    if args.limit_files == 0:
        with open('test.ddddd','wb') as fb:
            pickle.dump(train_data_set, fb)

print(len(test_data_set))
    
files_prepared = {}
class KerasBatchGenerator(object):
    # data_set contains train or test data_set
    # vocabin 
    def __init__(self, data_set):
        self.data_set = data_set
        self.ToCategorical = torch.eye(max_output)
            
    def generate(self):
        while True:
            #for py_program in self.data_set: 
            for python_file in self.data_set:
                #print(python_file)
                if not python_file in files_prepared:
                    py_program = tokenize.generate_tokens(open(python_file).readline)
                    full_python_file_string = []
                    for x in py_program:
                        #print(x)
                        transl = translator.translate(x)
                        if transl is not None:
                            full_python_file_string.append(transl)
                        #else:
                        #    print('and not used')
                    if args.enable_pretokenized_files:
                        with open(python_file+'.ddddd','wb') as fb:
                            pickle.dump(full_python_file_string, fb)
                        files_prepared[python_file] = 1
                else:
                     with open(python_file+'.ddddd','rb') as fb:
                         full_python_file_string = pickle.load(fb)
                position=0
                while position * args.max_length < len(full_python_file_string)-2:
                    end_is = (position+1)*args.max_length
                    if end_is >= len(full_python_file_string)-2:
                        end_is = len(full_python_file_string)-2
                    tmp_x = np.array([full_python_file_string[position*args.max_length:end_is]], dtype=int)
                    tmp_y = np.array([full_python_file_string[position*args.max_length+1:end_is+1]], dtype=int)
                    position += 1
                    ret = tmp_x, self.ToCategorical[tmp_y].reshape(1,-1,max_output) #wrong shape if exactly one is in
                    if len(ret[1].shape) != 3:
                        print(ret)
                    yield ret

train_data_generator = KerasBatchGenerator(train_data_set)
test_data_generator = KerasBatchGenerator(test_data_set)



input_dim = max_output
hidden_dim = max_output
n_layers = 1



batch_size = 1
seq_len = 1 # overwrite with more later

inp = torch.randn(batch_size, seq_len, input_dim)


l = train_data_generator.generate()
ii, oo = next(l)

print(ii.shape, oo.shape)

inp = train_data_generator.ToCategorical[ii].reshape(1,-1,max_output) 

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.register_buffer('hidden_state', torch.randn(n_layers, batch_size, hidden_dim))
        self.register_buffer('cell_state', torch.randn(n_layers, batch_size, hidden_dim))

    def forward(self, x):
        y, (self.hidden_state, self.cell_state) = self.lstm_layer(x, (self.hidden_state, self.cell_state))
        return y

# out, hidden2 = lstm_layer(inp, hidden)
net_model = Model()
out = net_model.forward(inp)

def error(pred, target): return ((pred-target)**2).mean()

loss = error(out, oo)

loss.backward()
print(net_model.lstm_layer.weight_ih_l0.grad)

net_model.lstm_layer.weight_ih_l0.grad.zero_()

print(net_model.lstm_layer.weight_ih_l0.grad)

for name, W in net_model.named_parameters(): print(name)