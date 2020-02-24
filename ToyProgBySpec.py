#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:53:25 2020

@author: detlef
"""

from tkinter import Tk, Canvas, mainloop, W



def getVariable(name):
    return name.getVar            

# takes a lambda function at the moment to compare two elements
    # e.g. between(ll,lambda a:ll[a].getVar('left')-ll[a-1].getVar('right'))
def between(ll, compare):
    ret=[]
    if len(ll)>1:
        for i in range(1,len(ll)):
            print(i,compare(i))
            ret.append(compare(i))
    return ret


# This is the main documented class, later classes are not documented for future syntax
class Character():
    def __init__(self):
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        #fixed content not changed by other variables
        self.TheCharacter = None
    
        #manageing variables (not used in later syntax)
        self.TheVars = {} #contains the variables from priority
        self.TheChanges = {} #contains the variables to be changed in priority during a reassignement
    
    def getVar(self, name):
        return self.TheVars[name]
    
    def setVar(self, name, value):
        self.TheVars[name]=value
    
    def restictions(self):
        # becomes zero for the correct values, uses priority to determine which to optimize
        # planed syntax would be:
        # top-bottom = height
        # right-left = width
        return [self.getVar('top')-self.getVar('bottom')-self.getVar('height'), 
                self.getVar('right')-self.getVar('left')-self.getVar('width')]
    
class Word():
    def __init__(self):
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        self.WordCharacters = []

        #manageing variables (not used in later syntax)
        self.TheVars = {} #contains the variables from priority
        self.TheChanges = {} #contains the variables to be changed in priority during a reassignement

    def restrictions(self):
        ret = []
        
        # this must get good syntax later !!!!
        if (self.WordCharacters.len>0):
            ret.append(self.WordCharacters[0].getVar('left')-self.getVar('left'))
            ll=self.WordCharacters
            ret += between(ll, lambda a: ll[a].getVar('left')-ll[a-1].getVar('right'))
            ret.append(self.WordCharacters[len(self.WordCharacters)-1].getVar('right')-self.TheVars('right'))
        return ret

class Line():
    def __init__(self):
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        self.LineWords = []

        #manageing variables (not used in later syntax)
        self.TheVars = {} #contains the variables from priority
        self.TheChanges = {} #contains the variables to be changed in priority during a reassignement

    def restrictions(self):
        ret = []
        
        # this must get good syntax later !!!!
        if (self.WordCharacters.len>0):
            ret.append(self.LineWords[0].getVar('left')-self.getVar('left'))
            ll=self.LineWords
            ret += between(ll, lambda a: ll[a].getVar('left')-ll[a-1].getVar('right'))
            ret.append(self.LineWords[len(self.WordCharacters)-1].getVar('right')-self.TheVars('right')) #too long must be managed here as allowed operation
        return ret

class Page():
    def __init__(self):
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        self.PageLines = []

        #manageing variables (not used in later syntax)
        self.TheVars = {} #contains the variables from priority
        self.TheChanges = {} #contains the variables to be changed in priority during a reassignement

    def restrictions(self):
        ret = []
        
        # this must get good syntax later !!!!
        if (self.WordCharacters.len>0):
            ret.append(self.LineWords[0].getVar('top')-self.getVar('top'))
            ll=self.PageLines
            ret += between(ll, lambda a: ll[a].getVar('top')-ll[a-1].getVar('bottom'))
            ret.append(self.PageLines[len(self.WordCharacters)-1].getVar('bottom')-self.TheVars('bottom')) #too long must be managed here as allowed operation
        return ret



#define operations between objects ?????????







# train lambda
a=Character()
a.setVar('left',3)
a.setVar('right',5)
b=Character()
b.setVar('left',6)
b.setVar('right',9)
c=Character()
c.setVar('left',15)
c.setVar('right',20)
ll=[a,b,c]
between(ll,lambda a:ll[a].getVar('left')-ll[a-1].getVar('right'))














# Toy Manager
master = Tk()
canvas_width = 800
canvas_height = 400
w = Canvas(master, width=canvas_width, height=canvas_height)
w.pack()

def click(event):
    print('button clicked',event)

def key(event):
    c = w.create_text(20, 30, anchor=W, font=("Times New Roman", int(25), "bold"),
            text=event.char)
    print('key pressed',event, 'bounding', w.bbox(c))

w.bind('<Button-1>', click)
master.bind('<Key>',key)
#mainloop()