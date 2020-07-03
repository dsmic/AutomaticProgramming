#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:04:58 2019

@author: detlef
"""
import pickle
import argparse
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.layers import Activation, Embedding, Dense,  Lambda, Input
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM
from tensorflow.keras.utils import to_categorical
import numpy as np
import tokenize
#from collections import OrderedDict
from threading import Lock
#from sortedcontainers import SortedList

#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # remove some optimization warnings for tensorflow

parser = argparse.ArgumentParser(description='train recurrent net.')
parser.add_argument('--lr', dest='lr',  type=float, default=1e-3)
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

parser.add_argument('--save_tokens', dest='save_tokens',  type=int, default=0)
parser.add_argument('--save_tokens_min_num', dest='save_tokens_min_num',  type=int, default=100)

args = parser.parse_args()


max_output = args.token_number

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
        
    def translate(self,token):
        # seems to be called by different threads?!
        with self.lock:
            if args.save_tokens != 0:
                if (token[0], token[1]) in saved_tokens:
                    return saved_tokens[(token[0], token[1])]
                if (token[0]) in saved_tokens:
                    return saved_tokens[(token[0])]
                if token[0] != 54: #ERRORtokens may appear
                    print('should not happen?', token, 'return', saved_pos)
                return saved_pos
            backok = False
            if args.remove_comments and (token[0] == tokenize.COMMENT or token[0] == tokenize.NL):
                #print('comment removed')
                return None
            if args.only_token_type or (args.only_token_detail and token[0] != tokenize.OP) or (args.only_token_detail_name and token[0] != tokenize.OP and token[0] != tokenize.NAME) or (args.no_string_details and token[0] == tokenize.STRING):
                used_part = (token[0]) # (type , string ) of the tokenizer
            else:
                used_part = (token[0],token[1]) # (type , string ) of the tokenizer
                backok =True
            #print(used_part)
            f = 1 - 1.0 / args.token_number
            for aa in self.used:
                self.used[aa] *= f
            #self.used.update((k, v * f) for k,v in self.used.items())
            if used_part in self.used:
                self.used_sorted.delete(used_part)
                self.used[used_part] += 1
                if self.used[used_part] > args.token_number / 2:
                    self.used[used_part] = args.token_number / 2
                #assert(self.used_sorted.check_order())
            else:
                self.used[used_part] = 1
            self.used_sorted.add(used_part)
            #print(token)
            self.calls += 1
            if used_part not in self.data:
                if len(self.free_numbers) == 0:
                    #oldest_old = min(self.used,key=self.used.get)
                    oldest = self.used_sorted.pop_lowest()
                    #assert(oldest == oldest_old)
                    self.free_numbers = [self.data[oldest]]
                    self.removed_numbers[self.data[oldest]] = 1
                    if args.debug:
                        print('deleted', oldest, self.used[oldest], self.data[oldest], len(self.removed_numbers))
                    #self.used_sorted.delete(oldest) already deleted with pop
                    del(self.used[oldest])
                    del(self.data[oldest])
                next_num_of_token = self.free_numbers[0]
                self.free_numbers=self.free_numbers[1:]
                self.data[used_part] = next_num_of_token
                if backok:
                    self.back[next_num_of_token] = used_part[1] # string of used part
                else:
                    if token.type == tokenize.NEWLINE:
                        self.back[next_num_of_token] = '\n'
                    else:
                        self.back[next_num_of_token] = "???"
            else:
                self.found += 1
            return self.data[used_part]

    def get_string(self, num_of_token):
        return self.back[num_of_token]
            
translator = Token_translate(max_output)            


def read_file(file_name):
    txt_file = open(file_name)
    return [line.strip('\n') for line in txt_file]

all_tokens = {}
def load_dataset(file_names, save_tokens = 0):
    global num_ord
    data_set = []
    count = 0
    py_programs = []    
    for file_name in file_names:
        count += 1
        if args.limit_files >0 and count > args.limit_files:
            break
        if save_tokens > 0 and count > save_tokens:
            break
        try:
            python_file = open(file_name)
            py_program = tokenize.generate_tokens(python_file.readline) # just check if no errors accure
            #list(py_program) #force tokenizing to check for errors
            #program_lines = []
            if save_tokens != 0:
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
saved_pos = 0    
if args.save_tokens != 0:
    load_dataset(read_file('python100k_train.txt_random'), args.save_tokens)
    print('used for counting',args.save_tokens, 'min num', args.save_tokens_min_num)
    for i in all_tokens:
        t = all_tokens[i]
        c = 0
        saved_tokens[(i)] = saved_pos
        saved_pos += 1
        for l in t:
            if t[l] > args.save_tokens_min_num:
                saved_tokens[(i,l)] = saved_pos
                saved_pos += 1
                c +=1
        print(i, len(all_tokens[i]), c, len(saved_tokens))
    max_output = saved_pos + 1
for l in saved_tokens:
    print(l, saved_tokens[l])
    
train_data_set = load_dataset(read_file('python100k_train.txt'+file_append))    

print(len(train_data_set))

test_data_set = load_dataset(read_file('python50k_eval.txt'+file_append))    

print(len(test_data_set))
    
files_prepared = {}
class KerasBatchGenerator(object):
    # data_set contains train or test data_set
    # vocabin 
    def __init__(self, data_set):
        self.data_set = data_set
            
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
                model.reset_states()
#                if args.debug:
#                    print("\nnext file used "+ python_file.strip())
                while position * args.max_length < len(full_python_file_string)-2:
                    end_is = (position+1)*args.max_length
                    if end_is >= len(full_python_file_string)-2:
                        end_is = len(full_python_file_string)-2
                    tmp_x = np.array([full_python_file_string[position*args.max_length:end_is]], dtype=int)
                    tmp_y = np.array([full_python_file_string[position*args.max_length+1:end_is+1]], dtype=int)
                    position += 1
                    ret = tmp_x, to_categorical(tmp_y, num_classes=max_output).reshape(1,-1,max_output) #wrong shape if exactly one is in
                    if len(ret[1].shape) != 3:
                        print(ret)
                    yield ret

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
    embed_len = max_output
if args.pretrained_name is not None:
  from tensorflow.keras.models import load_model
  model = load_model(args.pretrained_name)
else:
  inputs = Input(batch_shape=(1,None,))
  embeds = Embedding(max_output, embed_len, embeddings_initializer='identity', trainable=args.embeddings_trainable)(inputs)
  lstm1 = LSTM_use(args.hidden_size, return_sequences=True, stateful = True)(embeds)
  if args.attention:
    lstm1b = Lambda(attentions_layer)(lstm1)
  else:
    lstm1b = lstm1
  if args.two_LSTM:
      lstm4 = LSTM_use(args.hidden_size, return_sequences=True, stateful = True)(lstm1b)
  else:
      lstm4 = lstm1b

  x = Dense(max_output)(lstm4)
  predictions = Activation('softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)
print(model.summary())

if args.load_weights_name:
    model.load_weights(args.load_weights_name, by_name=True)
    print('weights loaded')

if args.benchmark_parsing > 0:
    performance_test = train_data_generator.generate()
    for i in range(args.benchmark_parsing):
        next(performance_test)
        print(i)
    import sys
    sys.exit()

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
model.fit_generator(train_data_generator.generate(), args.epoch_size, args.epochs, 
                    validation_data=test_data_generator.generate(), validation_steps=args.epoch_size / 10, 
                    callbacks=[checkpointer, tensorboard, terminate_on_key])

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

print(len(translator.free_numbers))
print(translator.found / translator.calls, translator.calls)
save_dict_to_file(translator.back, args.final_name+'_back')
save_dict_to_file(translator.used, args.final_name+'_used')
save_dict_to_file(translator.data, args.final_name+'_data')
save_dict_to_file(translator.free_numbers, args.final_name+'_free_numbers')



