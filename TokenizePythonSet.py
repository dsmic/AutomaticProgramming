#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:08:17 2019

@author: detlef
"""
import tokenize


class Token_translate:
    def __init__(self, num_free):
        self.data = {}
        self.used = {}
        self.back = {}
        self.free_numbers = [i for i in range(num_free)] 
    
    def translate(self,token):
        used_part = (token[0],token[1]) # (type , string ) of the tokenizer
        for all in self.used:
            self.used[all] *= 0.99
        self.used[used_part] = 1
        if used_part not in self.data:
            if len(self.free_numbers) == 0:
                oldest = min(self.used,key=self.used.get)
                self.free_numbers.append(self.data[oldest])
                del(self.used[oldest])
                del(self.data[oldest])
            next_num_of_token = self.free_numbers[0]
            self.free_numbers=self.free_numbers[1:]
            self.data[used_part] = next_num_of_token
            self.back[next_num_of_token] = used_part[1] # string of used part
        
        print(len(self.data), len(self.used), len(self.free_numbers))
        return self.data[used_part]

    def get_string(self, num_of_token):
        return self.back[num_of_token]

            
translator = Token_translate(1000)            
            


def read_file(file_name):
    txt_file = open(file_name)
    return [line.strip('\n') for line in txt_file]

def load_dataset(file_names):
    global num_ord
    data_set = []
    for file_name in file_names:
        try:
            python_file = open(file_name)
            py_program = tokenize.generate_tokens(python_file.readline)
            #program_lines = []
            for py_token in py_program:
                print(py_token)
                token_number = translator.translate(py_token)
                print("---",token_number, '- ' + translator.get_string(token_number)+' -')

        except UnicodeDecodeError as e:
            print(file_name + '\n  wrong encoding ' + str(e))
    return data_set

train_data_set = load_dataset(read_file('python100k_train.txt'))    
