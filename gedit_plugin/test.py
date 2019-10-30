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

import sys, os
current_line = int(os.getenv('GEDIT_CURRENT_LINE_NUMBER'))
current_offset = int(os.getenv('GEDIT_CURRENT_LINE_OFFSET'))

from tensorflow.keras.models import load_model
import numpy as np

model = load_model('/home/detlef/AutomaticProgramming/final_model.hdf5')

pgm_lines = []
ln=0
for line in sys.stdin:
    ln += 1
    if ln == current_line:
      pgm_lines.append(line[:current_offset-1])
      break
    else:
      pgm_lines.append(line)

for line in pgm_lines:
    print(line.strip('\n'))

def load_dict_from_file(file_name):
    f = open(file_name+'.dict','r')
    data=f.read()
    f.close()
    return eval(data)

used_ords = load_dict_from_file('/home/detlef/AutomaticProgramming/final_model')
used_rev = {}
for x,y in used_ords.items():
  if x == '\n':
    used_rev[y] = '\n'
  else:
    used_rev[y] = chr(x)
    #print(y,chr(x))

def tr_char(inp):
    if ord(inp) in used_ords:
      return used_ords[ord(inp)]
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! char not found !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return 0
 
full_python_file_string = []
for pgm_line in pgm_lines:
    program_line = [tr_char(char_in_line) for char_in_line in pgm_line.strip('\n')]
    full_python_file_string.extend(program_line)
    full_python_file_string.append(0)

full_python_file_string.append(0)
full_python_file_string.append(4)
full_python_file_string.append(5)

inn = np.array([full_python_file_string], dtype=int)
prediction = model.predict(inn)[0]
print(prediction)

def list_to_string(prediction):
    s=""
    for i in range(prediction.shape[0]):
        #print(np.argmax(prediction[i]))
        s += used_rev[np.argmax(prediction[i])]
    return s

p_str = list_to_string(prediction)
print('--------------------------')
print(p_str)
print('--------------------------')

