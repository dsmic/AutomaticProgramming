#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:04:58 2019

@author: detlef
"""

def read_file(file_name):
    txt_file = open(file_name)
    return [line.strip('\n') for line in txt_file]
    
used_ords = {}
num_ord = 0 # starting the translation table

def load_dataset(file_names):
    global num_ord
    data_set = []
    for line in file_names:
        try:
            py_program = read_file(line.strip())
            program_lines = []
            for t2 in py_program:
                program_chars = []
                for t3 in t2:
                    if ord(t3) not in used_ords:
                        used_ords[ord(t3)] = num_ord
                        num_ord += 1
                    program_chars.append(used_ords[ord(t3)])
                program_lines.append(program_chars)
            data_set.append(program_lines)
        except UnicodeDecodeError as e:
            print(line + '\n  wrong encoding ' + str(e))
    return data_set

train_file_names = read_file('python100k_train.txt')
train_data_set = load_dataset(train_file_names)    

test_file_names = read_file('python50k_eval.txt')
test_data_set = load_dataset(test_file_names)    

print(len(train_data_set))
print(len(used_ords))
print(num_ord)
print(len(test_data_set))
print(len(used_ords))
print(num_ord)