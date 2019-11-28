# gedit external program
# #!/bin/sh
# source /home/detlef/tf2-gpu/bin/activate
# python /home/detlef/AutomaticProgramming/gedit_plugin/test.py 

# external tool plugin: in functions.py added the GEDIT_CURRENT_LINE_OFFSET
#
# Current line number
#        piter = document.get_iter_at_mark(document.get_insert())
#        capture.set_env(GEDIT_CURRENT_LINE_NUMBER=str(piter.get_line() + 1))
#        capture.set_env(GEDIT_CURRENT_LINE_OFFSET=str(piter.get_line_offset() + 1))

import sys, os, tokenize, io
current_line = int(os.getenv('GEDIT_CURRENT_LINE_NUMBER'))
current_offset = int(os.getenv('GEDIT_CURRENT_LINE_OFFSET'))

args_remove_comments = True
args_only_token_type = False
args_only_token_detail_name = True
args_only_token_detail = False
args_token_number = 4000
max_output = args_token_number


from tensorflow.keras.models import load_model
import numpy as np

model = load_model('/home/detlef/AutomaticProgramming/final_model.hdf5')

pgm_lines = ""
ln=0
for line in sys.stdin:
    ln += 1
    if ln == current_line:
      pgm_lines +=line[:current_offset-1]
      break
    else:
      pgm_lines += line

print(pgm_lines)

string_file = io.StringIO(pgm_lines)

print(string_file)

def load_dict_from_file(file_name):
    f = open(file_name+'.dict','r')
    data=f.read()
    f.close()
    return eval(data)

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
        #self.used_sorted = token_sort(self.used)
        self.back = {}
        self.free_numbers = [i for i in range(num_free)] 
#        self.lock = Lock()
        self.found = 0
        self.calls = 0
        
    def translate(self,token):
        # seems to be called by different threads?!
#        with self.lock:
            backok = False
            if args_remove_comments and (token[0] == tokenize.COMMENT or token[0] == tokenize.NL):
                #print('comment removed')
                return None
            if args_only_token_type or (args_only_token_detail and token[0] != tokenize.OP) or (args_only_token_detail_name and token[0] != tokenize.OP and token[0] != tokenize.NAME):
                used_part = (token[0]) # (type , string ) of the tokenizer
            else:
                used_part = (token[0],token[1]) # (type , string ) of the tokenizer
                backok =True
            #print(used_part)
            f = 1 - 1.0 / args_token_number
            for aa in self.used:
                self.used[aa] *= f
            #self.used.update((k, v * f) for k,v in self.used.items())
            if used_part in self.used:
                #self.used_sorted.delete(used_part)
                self.used[used_part] += 1
                if self.used[used_part] > args_token_number / 10:
                    self.used[used_part] = args_token_number / 10
                #assert(self.used_sorted.check_order())
            else:
                self.used[used_part] = 1
            #self.used_sorted.add(used_part)
            self.calls += 1
            if used_part not in self.data:
                if len(self.free_numbers) == 0:
                    oldest = min(self.used,key=self.used.get)
                    #oldest = self.used_sorted.pop_lowest()
                    #assert(oldest == oldest_old)
                    self.free_numbers = [self.data[oldest]]
                    #if args.debug:
                    #    print('deleted', oldest, self.used[oldest])
                    #self.used_sorted.delete(oldest) already deleted with pop
                    del(self.used[oldest])
                    del(self.data[oldest])
                next_num_of_token = self.free_numbers[0]
                self.free_numbers=self.free_numbers[1:]
                self.data[used_part] = next_num_of_token
                if backok:
                    self.back[next_num_of_token] = used_part[1] # string of used part
                else:
                    self.back[next_num_of_token] = "???"
            else:
                self.found += 1
            return self.data[used_part]

    def get_string(self, num_of_token):
        return self.back[num_of_token]
            
translator = Token_translate(max_output)            
translator.back = load_dict_from_file('/home/detlef/AutomaticProgramming/final_model_back')
translator.used = load_dict_from_file('/home/detlef/AutomaticProgramming/final_model_used')
translator.data = load_dict_from_file('/home/detlef/AutomaticProgramming/final_model_data')
translator.free_numbers = load_dict_from_file('/home/detlef/AutomaticProgramming/final_model_free_numbers')

tokens = tokenize.generate_tokens(string_file.readline)

#print([t for t in tokens])

full_python_file_string = [translator.translate(x) for x in tokens]
# for pgm_line in pgm_lines:
#     program_line = [tr_char(char_in_line) for char_in_line in pgm_line.strip('\n')]
#     full_python_file_string.extend(program_line)
#     full_python_file_string.append(0)
print(full_python_file_string)
inn = np.array([full_python_file_string], dtype=int)
prediction = model.predict(inn)[0]
print(prediction)

def list_to_string(prediction):
    s=""
    for i in range(prediction.shape[0]):
        #print(np.argmax(prediction[i]))
        s += translator.back[np.argmax(prediction[i])]
    return s

p_str = list_to_string(prediction)
print('--------------------------')
print(p_str)
print('--------------------------')

