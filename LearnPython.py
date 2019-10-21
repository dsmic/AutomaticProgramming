#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:04:58 2019

@author: detlef
"""

import argparse
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Embedding, Dense,  Lambda, Input
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM
from tensorflow.keras.utils import to_categorical

parser = argparse.ArgumentParser(description='train recurrent net.')
parser.add_argument('--lr', dest='lr',  type=float, default=1e-3)
parser.add_argument('--epochs', dest='epochs',  type=int, default=50)
parser.add_argument('--hidden_size', dest='hidden_size',  type=int, default=50)
parser.add_argument('--final_name', dest='final_name',  type=str, default='final_model')
parser.add_argument('--pretrained_name', dest='pretrained_name',  type=str, default=None)
parser.add_argument('--attention', dest='attention', action='store_true')
parser.add_argument('--depth', dest='depth',  type=int, default=3)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--only_one', dest='only_one', action='store_true')
parser.add_argument('--revert', dest='revert', action='store_true')
parser.add_argument('--add_history', dest='add_history', action='store_true')
parser.add_argument('--RNN_type', dest='RNN_type',  type=str, default='LSTM')
parser.add_argument('--gpu_mem', dest='gpu_mem',  type=float, default=1)
parser.add_argument('--fill_vars_with_atoms', dest='fill_vars_with_atoms', action='store_true')
parser.add_argument('--rand_atoms', dest='rand_atoms', action='store_true')
parser.add_argument('--float_type', dest='float_type',  type=str, default='float32')

parser.add_argument('--limit_files', dest='limit_files',  type=int, default=0)

args = parser.parse_args()

RNN_type = {}
  

RNN_type['LSTM'] = LSTM
RNN_type['GRU'] = GRU
RNN_type['SimpleRNN'] = SimpleRNN

LSTM_use = RNN_type[args.RNN_type]

def read_file(file_name):
    txt_file = open(file_name)
    return [line.strip('\n') for line in txt_file]
    
used_ords = {}
used_ords['\n']=0
num_ord = [1] # starting the translation table 0 reserved to seperate lines which are striped here

def load_dataset(file_names):
    global num_ord
    data_set = []
    count_files = 0
    for line in file_names:
        if args.limit_files > 0:
            if count_files > args.limit_files:
                break
            count_files +=1
        try:
            py_program = read_file(line.strip())
            program_lines = []
            for t2 in py_program:
                program_chars = []
                for t3 in t2:
                    if ord(t3) not in used_ords:
                        used_ords[ord(t3)] = num_ord[0]
                        num_ord[0] += 1
                    program_chars.append(used_ords[ord(t3)])
                program_lines.append(program_chars)
            data_set.append(program_lines)
        except UnicodeDecodeError as e:
            print(line + '\n  wrong encoding ' + str(e))
    return data_set

train_data_set = load_dataset(read_file('python100k_train.txt'))    


print(len(train_data_set))
print(len(used_ords))
print(num_ord[0])

test_data_set = load_dataset(read_file('python50k_eval.txt'))    


print(len(test_data_set))
print(len(used_ords))
print(num_ord[0])


max_output = len(used_ords)

class KerasBatchGenerator(object):
    # data_set contains train or test data_set
    # vocabin 
    def __init__(self, data_set):
        self.data_set = data_set
            
    def generate(self):
        while True:
            for python_file in self.data_set:
                full_python_file_string = []
                for python_line in python_file:
                    full_python_file_string.extend(python_line)
                    full_python_file_string.append(0)
                    
                yield full_python_file_string, to_categorical(full_python_file_string, num_classes=max_output)


train_data_generate = KerasBatchGenerator(test_data_set).generate()
count=0
for tt in train_data_generate:
    print(tt)
    count+=1
    if count>10:
        break


###################################################################
# Network


def attentions_layer(x):
  from keras import backend as K
  x1 = x[:,:,1:]
  x2 = x[:,:,0:1]
  x2 = K.softmax(x2)

  x=x1*x2

  return x

hidden_size = args.hidden_size

max_length = 1000
if args.pretrained_name is not None:
  from keras.models import load_model
  model = load_model(args.pretrained_name)
  print("loaded model",model.layers[0].input_shape[1])
  ml = model.layers[0].input_shape[1]
  if (ml != max_length):
    print("model length",ml,"different from data length",max_length)
    max_length = ml
else:
  inputs = Input(shape=(None,))
  embeds = Embedding(len(used_ords), len(used_ords), embeddings_initializer='identity', trainable=True)(inputs)
  lstm1 = LSTM_use(hidden_size, return_sequences=True)(embeds)
  if args.attention:
    lstm1b = Lambda(attentions_layer)(lstm1)
  else:
    lstm1b = lstm1
  lstm4 = LSTM_use(hidden_size, return_sequences=False)(lstm1b)

  x = Dense(max_output +1)(lstm4)
  predictions = Activation('softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)


