#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:22:15 2020

@author: detlef
"""

import numpy as np
import tokenize

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
                data_set.append(key)
            count += 1
        except UnicodeDecodeError as e:
            print(file_name + '\n  wrong encoding ' + str(e))
        except tokenize.TokenError as e:
            print(file_name + '\n  token error ' + str(e))
    return data_set


train_data_set = load_dataset(read_file('python100k_train.txt'))    
