#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:04:58 2019

@author: detlef
"""

import argparse
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Embedding, Dense,  Lambda, Input
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM
from tensorflow.keras.utils import to_categorical
import numpy as np
#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # remove some optimization warnings for tensorflow

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
parser.add_argument('--RNN_type', dest='RNN_type',  type=str, default='GRU')
parser.add_argument('--gpu_mem', dest='gpu_mem',  type=float, default=1)
parser.add_argument('--fill_vars_with_atoms', dest='fill_vars_with_atoms', action='store_true')
parser.add_argument('--rand_atoms', dest='rand_atoms', action='store_true')
parser.add_argument('--float_type', dest='float_type',  type=str, default='float32')

parser.add_argument('--limit_files', dest='limit_files',  type=int, default=0)
parser.add_argument('--epoch_size', dest='epoch_size',  type=int, default=1000)
parser.add_argument('--max_length', dest='max_length',  type=int, default=20000)

args = parser.parse_args()

RNN_type = {}
  

RNN_type['LSTM'] = LSTM
RNN_type['GRU'] = GRU
RNN_type['SimpleRNN'] = SimpleRNN

LSTM_use = RNN_type[args.RNN_type]

if tf.__version__ < "2.0":
    from tensorflow.keras.backend import set_session
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
else:
    #tensorflow 2.0 sets memory growth per default
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

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
    for filename in file_names:
        if args.limit_files > 0:
            if count_files > args.limit_files:
                break
            count_files +=1
        try:
            py_program = read_file(filename.strip())
            program_lines = []
            for t2 in py_program:
                program_chars = []
                for t3 in t2:
                    if ord(t3) not in used_ords:
                        used_ords[ord(t3)] = num_ord[0]
                        num_ord[0] += 1
                    program_chars.append(used_ords[ord(t3)])
                program_lines.append(program_chars)
            data_set.append(filename)
        except UnicodeDecodeError as e:
            print(filename + '\n  wrong encoding ' + str(e))
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
                py_program = read_file(python_file.strip())
                program_lines = []
                len_of_file = 0
                for t2 in py_program:
                    program_chars = []
                    for t3 in t2:
                        if ord(t3) not in used_ords:
                            used_ords[ord(t3)] = num_ord[0]
                            num_ord[0] += 1
                        program_chars.append(used_ords[ord(t3)])
                    program_lines.append(program_chars)
                    len_of_file += len(program_chars)
#                    if args.max_length < len_of_file:
#                        break
                full_python_file_string = []
                for python_line in program_lines:
                    full_python_file_string.extend(python_line)
                    full_python_file_string.append(0)
#                    if args.max_length < len(full_python_file_string):
#                        break
#                if args.max_length < len(full_python_file_string):
#                    print('\n'+str(len(full_python_file_string)) + ' character python file cut, as it was longer than '+str(args.max_length))
#                    full_python_file_string= full_python_file_string[:args.max_length]
#                    model.reset_states()
                position=0
                model.reset_states()
                while position * args.max_length < len(full_python_file_string):
                    tmp_x = np.array([full_python_file_string[position*args.max_length:(position+1)*args.max_length]], dtype=int)
                    tmp_y = np.array([full_python_file_string[position*args.max_length:(position+1)*args.max_length]], dtype=int)
                    yield tmp_x, to_categorical(tmp_y, num_classes=max_output)


train_data_generator = KerasBatchGenerator(train_data_set)
test_data_generator = KerasBatchGenerator(test_data_set)
# count=0
# for tt in train_data_generator.generate():
# #    print(tt)
#     count+=1
# #    if count>10:
# #        break

# print(count)

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

if args.pretrained_name is not None:
  from keras.models import load_model
  model = load_model(args.pretrained_name)
  print("loaded model",model.layers[0].input_shape[1])
  ml = model.layers[0].input_shape[1]
  if (ml != args.max_length):
    print("model length",ml,"different from data length", args.max_length)
    args.max_length = ml
else:
  inputs = Input(batch_shape=(1,None,))
  embeds = Embedding(len(used_ords), len(used_ords), embeddings_initializer='identity', trainable=True)(inputs)
  lstm1 = LSTM_use(hidden_size, return_sequences=True, stateful = True)(embeds)
  if args.attention:
    lstm1b = Lambda(attentions_layer)(lstm1)
  else:
    lstm1b = lstm1
  lstm4 = LSTM_use(hidden_size, return_sequences=True, stateful = True)(lstm1b)

  x = Dense(max_output)(lstm4)
  predictions = Activation('softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)

print("starting",args)
#checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)

num_epochs = args.epochs

model.compile(loss='categorical_crossentropy', optimizer = 'SGD', metrics=['categorical_accuracy'])
model.fit_generator(train_data_generator.generate(), args.epoch_size, num_epochs, validation_data=test_data_generator.generate(), validation_steps=args.epoch_size / 10) 





