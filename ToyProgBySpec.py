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



class BaseRules():
    def __init__(self):
        #manageing variables (not used in later syntax)
        self.TheVars = {} #contains the variables from priority
        self.TheChanges = {} #contains the variables to be changed in priority during a reassignement
        self.childs = []
        
    def getVar(self, name):
        return self.TheVars[name]
    
    def setVar(self, name, value):
        self.TheVars[name]=value

    def takeToMuch(self, _):
        raise ValueError('Can not take Elements from before.')

# This is the main documented class, later classes are not documented for future syntax
class Character(BaseRules):
    def __init__(self, ch):
        BaseRules.__init__(self)
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        #fixed content not changed by other variables
        self.TheCharacter = ch
    
    def restictions(self):
        # becomes zero for the correct values, uses priority to determine which to optimize
        # planed syntax would be:
        # top-bottom = height
        # right-left = width
        return [self.getVar('top')-self.getVar('bottom')-self.getVar('height'), 
                self.getVar('right')-self.getVar('left')-self.getVar('width')], None
    
class Word(BaseRules):
    def __init__(self):
        BaseRules.__init__(self)
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        self.WordCharacters = self.childs

    def addCharacter(self, ch):
        l=Character(ch)
        self.WordCharacters.append(l)

    def restrictions(self):
        ret = []
        
        # this must get good syntax later !!!!
        if (self.WordCharacters.len>0):
            ret.append(self.WordCharacters[0].getVar('left')-self.getVar('left'))
            ll=self.WordCharacters
            ret += between(ll, lambda a: ll[a].getVar('left')-ll[a-1].getVar('right'))
            ret.append(self.WordCharacters[len(self.WordCharacters)-1].getVar('right')-self.getVar('right'))
        return ret, None

class Line(BaseRules):
    def __init__(self):
        BaseRules.__init__(self)
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        self.LineWords = self.childs

    def takeToMuch(self, ToMuch):
        self.LineWords.insert[0,ToMuch]

    def addWord(self):
        l=Word()
        self.LineWords.append(l)
        return l

    def restrictions(self):
        ret = []
        ToLong = None
        # this must get good syntax later !!!!
        if (self.WordCharacters.len>0):
            ret.append(self.LineWords[0].getVar('left')-self.getVar('left'))
            ll=self.LineWords
            ret += between(ll, lambda a: ll[a].getVar('left')-ll[a-1].getVar('right'))
            if self.LineWords[len(self.WordCharacters)-1].getVar('right')>self.getVar('right'):
                #too long must be managed here as allowed operation
                ToLong = self.lineWords.pop()
            ret.append(min([l.getVar('top') for l in ll])-self.getVar('top'))
            ret.append(max([l.getVar('bottom') for l in ll])-self.getVar('bottom'))
            
        return ret, ToLong

class Page(BaseRules):
    def __init__(self):
        BaseRules.__init__(self)
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        self.PageLines = self.childs
    
    def takeToMuch(self, ToMuch):
        self.PageLines.insert[0,ToMuch]
    
    def addLine(self):
        l=Line()
        self.PageLines.append(l)
        return l
        
    def restrictions(self):
        ret = []
        ToLong = None
        # this must get good syntax later !!!!
        if (self.WordCharacters.len>0):
            ret.append(self.PageLines[0].getVar('top')-self.getVar('top'))
            ret.append(self.PageLines[0].getVar('left')-self.getVar('left'))
            ret.append(self.PageLines[0].getVar('right')-self.getVar('right'))
            
            ll=self.PageLines
            ret += between(ll, lambda a: ll[a].getVar('top')-ll[a-1].getVar('bottom'))
            if self.PageLines[len(self.WordCharacters)-1].getVar('bottom') > self.getVar('bottom'):
                #too long must be managed here as allowed operation
                ToLong = self.PageLines.pop()
        return ret, ToLong



# define operations between objects ?????????
# takeToMuch can take from before, restrictions can return ToLong







# train lambda
a=Character('a')
a.setVar('left',3)
a.setVar('right',5)
b=Character('b')
b.setVar('left',6)
b.setVar('right',9)
c=Character('c')
c.setVar('left',15)
c.setVar('right',20)
ll=[a,b,c]
between(ll,lambda a:ll[a].getVar('left')-ll[a-1].getVar('right'))


#a.takeToMuch(None)

testpage = Page()
testpage.setVar('left',0)
testpage.setVar('top',0)
testpage.setVar('left',0)
testpage.setVar('right',0)

actualLine = testpage.addLine()
actualWord = actualLine.addWord()












# Toy Manager
master = Tk()
canvas_width = 800
canvas_height = 400
w = Canvas(master, width=canvas_width, height=canvas_height)
w.pack()

def click(event):
    print('button clicked',event)

def key(event):
    global actualWord
    c = w.create_text(20, 30, anchor=W, font=("Times New Roman", int(25), "bold"),
            text=event.char)
    print('key pressed',event, 'bounding', w.bbox(c))
    ch=event.char
    if ch== ' ':
        actualWord = actualLine.addWord()
    else:
        actualWord.addCharacter(ch)
        
w.bind('<Button-1>', click)
master.bind('<Key>',key)
mainloop()