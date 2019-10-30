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

ln=0
for line in sys.stdin:
    ln += 1
    if ln == current_line:
      print(line[:current_offset])
    else:
      print(line)
    if ln >= current_line:
      break


