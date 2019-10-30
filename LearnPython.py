#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:04:58 2019

@author: detlef
"""

import argparse
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
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
args = parser.parse_args()

RNN_type = {}
RNN_type['LSTM'] = LSTM
RNN_type['GRU'] = GRU
RNN_type['SimpleRNN'] = SimpleRNN

LSTM_use = RNN_type[args.RNN_type]

#tensorflow 2.0b sets memory growth per default, seems to be changed ?!
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
used_ords[ord('\n')]=0 
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
            #program_lines = []
            for t2 in py_program:
                #program_chars = []
                for t3 in t2:
                    if ord(t3) not in used_ords:
                        used_ords[ord(t3)] = num_ord[0]
                        num_ord[0] += 1
                    #program_chars.append(used_ords[ord(t3)])
                #program_lines.append(program_chars)
            data_set.append(filename)
        except UnicodeDecodeError as e:
            print(filename + '\n  wrong encoding ' + str(e))
    return data_set

if args.rand_files:
    file_append="_random"
else:
    file_append=""
train_data_set = load_dataset(read_file('python100k_train.txt'+file_append))    

print(len(train_data_set))
print(len(used_ords))
print(num_ord[0])

test_data_set = load_dataset(read_file('python50k_eval.txt'+file_append))    

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
                full_python_file_string = []
                for pgm_line in py_program:
                    program_line = [used_ords[ord(char_in_line)] for char_in_line in pgm_line]
                    full_python_file_string.extend(program_line)
                    full_python_file_string.append(0)
                position=0
                model.reset_states()
                if args.debug:
                    print("\nnext file used "+ python_file.strip())
                while position * args.max_length < len(full_python_file_string)-1:
                    end_is = (position+1)*args.max_length
                    if end_is >= len(full_python_file_string)-1:
                        end_is = len(full_python_file_string)-1
                    tmp_x = np.array([full_python_file_string[position*args.max_length:end_is]], dtype=int)
                    tmp_y = np.array([full_python_file_string[position*args.max_length+1:end_is+1]], dtype=int)
                    position += 1
                    yield tmp_x, to_categorical(tmp_y, num_classes=max_output)

train_data_generator = KerasBatchGenerator(train_data_set)
test_data_generator = KerasBatchGenerator(test_data_set)

def attentions_layer(x):
  from keras import backend as K
  x1 = x[:,:,1:]
  x2 = x[:,:,0:1]
  x2 = K.softmax(x2)

  x=x1*x2

  return x

# at the moment loaded models seem not to support cudnn
# https://github.com/tensorflow/tensorflow/issues/33601
# you can use load_weights_name to load the weights into the model
  
if args.embed_len is not None:
    embed_len = args.embed_len
else:
    embed_len = len(used_ords)
if args.pretrained_name is not None:
  from tensorflow.keras.models import load_model
  model = load_model(args.pretrained_name)
else:
  inputs = Input(batch_shape=(1,None,))
  embeds = Embedding(len(used_ords), embed_len, embeddings_initializer='identity', trainable=args.embeddings_trainable)(inputs)
  lstm1 = LSTM_use(args.hidden_size, return_sequences=True, stateful = True)(embeds)
  if args.attention:
    lstm1b = Lambda(attentions_layer)(lstm1)
  else:
    lstm1b = lstm1
  lstm4 = LSTM_use(args.hidden_size, return_sequences=True, stateful = True)(lstm1b)

  x = Dense(max_output)(lstm4)
  predictions = Activation('softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)
print(model.summary())

if args.load_weights_name:
    model.load_weights(args.load_weights_name, by_name=True)
    print('weights loaded')

print("starting",args)
import os
class TerminateKey(Callback):
    def on_epoch_end(self, batch, logs=None):
        if os.path.exists(args.EarlyStop):
            self.model.stop_training = True

terminate_on_key = TerminateKey()

tensorboard = TensorBoard(log_dir = args.tensorboard_logdir)

checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)

model.compile(loss='categorical_crossentropy', optimizer = 'SGD', metrics=['categorical_accuracy'])
model.fit_generator(train_data_generator.generate(), args.epoch_size, args.epochs, validation_data=test_data_generator.generate(), validation_steps=args.epoch_size / 10, callbacks=[checkpointer, tensorboard, terminate_on_key])

if os.path.exists(args.EarlyStop) and os.path.getsize(args.EarlyStop)==0:
    os.remove(args.EarlyStop)
    print('removed',args.EarlyStop)

model.save(args.final_name+'.hdf5')
model.save_weights(args.final_name+'-weights.hdf5')

def save_dict_to_file(dic, file_name):
    f = open(file_name+'.dict','w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(file_name):
    f = open(file_name+'.dict','r')
    data=f.read()
    f.close()
    return eval(data)

save_dict_to_file(used_ords,args.final_name)



