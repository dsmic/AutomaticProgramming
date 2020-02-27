#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:53:25 2020

@author: detlef
"""

from tkinter import Tk, Canvas, mainloop, W
import numpy as np
import operator

def getVariable(name):
    return name.getVar            

# takes a lambda function at the moment to compare two elements
    # e.g. between(ll,lambda a:ll[a].getVar('left')-ll[a-1].getVar('right'))
def for_all(ll, compare):
    ret=[]
    for i in range(0,len(ll)):
        ret.append(compare(i))
    return ret

def between(ll, compare):
    ret=[]
    for i in range(1,len(ll)):
        ret.append(compare(i))
    return ret



class BaseRules():
    def draw(self):
        print(self, 'char nodraw', round(self.getVar('left')), round(self.getVar('top')), round(self.getVar('right')), round(self.getVar('bottom')))
        pass
    
    def __init__(self):
        #manageing variables (not used in later syntax)
        self.TheVars = {} #contains the variables from priority
        for l in self.priority:
            self.setVar(l,0)
            
        self.TheChanges = {} #contains the variables to be changed in priority during a reassignement
        self.childs = []
        self.not_optimizing = None
            
    def getVar(self, name):
        return self.TheVars[name]
    
    def setVar(self, name, value):
        self.TheVars[name]=value

    def takeToMuch(self, _):
        raise ValueError('Can not take Elements from before.')
    
    def full_restrictions(self, debug = 0):
        ret = self.restrictions()
        #print('r',ret)
        for c in self.childs:
            ret += c.full_restrictions(debug=debug)
        if debug:
            print("full", type(self).__name__, len(ret), ret)
        return ret
    
    def get_all_self_and_childs(self):
        ret = [self]
        for c in self.childs:
            ret += c.get_all_self_and_childs()
        return ret
        
    def optimize_jakobi(self):
        jakobi_list = []
        before = self.full_restrictions()
        all_objects = self.get_all_self_and_childs()
        for obj in all_objects:
            opt_vars = obj.priority
            for vv in opt_vars:
                tmp = obj.getVar(vv)
                obj.setVar(vv, tmp + 1)
                after = self.full_restrictions()
                diff = list(map(operator.sub, after, before))
                obj.setVar(vv, tmp)
                jakobi_list.append(diff)
        f_x = np.array(before)
        JK = np.array(jakobi_list)
        JK_t = JK.transpose()
        JK_t_i = np.linalg.pinv(JK_t, rcond=0.0001)
        delta = np.dot(JK_t_i, f_x)
        print(delta)
        i=0
        for obj in all_objects:
            opt_vars = obj.priority
            for vv in opt_vars:
                print(vv, round(obj.getVar(vv)), round(delta[i]), round(obj.getVar(vv) - delta[i]))
                obj.setVar(vv, obj.getVar(vv) - delta[i])
                i += 1
                
        
    def optimize(self):
        self.optimize_jakobi()
        return
        #replace with multi dim newton !! Jakobi and invert
        all_objects = self.get_all_self_and_childs()
        for obj in all_objects:
            opt_vars = obj.priority
            opt_vars.reverse()
            for vv in opt_vars:
                before = self.full_restrictions()
                obj.setVar(vv, obj.getVar(vv)+1)
                after = self.full_restrictions()
                diff = list(map(operator.sub, after, before))
                correct = 0
                num_correct = 0
                bb = before[:]
                for l in diff:
                    dd = bb.pop(0)
                    if l != 0:
                        num_correct += 1
                        correct +=-dd/l
                
                if num_correct > 0: correct /= num_correct
                #correct *= 0.9
                correct -= 1        
                obj.setVar(vv, obj.getVar(vv)+correct)
                #if correct != -1:
                #    print(obj, vv, self.full_restrictions(), correct + 1)

        
                
                
# This is the main documented class, later classes are not documented for future syntax
class Character(BaseRules):
    def draw(self):
            print(self, 'char draw', self.TheCharacter, round(self.getVar('left')), round(self.getVar('top')), round(self.getVar('right')), round(self.getVar('bottom')))
            w.create_text(self.getVar('left'), self.getVar('top'), anchor=W, font=("Times New Roman", int(25), "bold"),
            text=self.TheCharacter)

    def __init__(self, ch):
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved
        self.not_optimizing = ['height', 'width']
        
        #fixed content not changed by other variables
        self.TheCharacter = ch
        BaseRules.__init__(self)
        
    def restrictions(self):
        # becomes zero for the correct values, uses priority to determine which to optimize
        # planed syntax would be:
        # top-bottom = height
        # right-left = width
        return [self.getVar('bottom')-self.getVar('top')-self.getVar('height'), 
                self.getVar('right')-self.getVar('left')-self.getVar('width')]
    
class Word(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        BaseRules.__init__(self)
        self.WordCharacters = self.childs
        
    def addCharacter(self, ch):
        l=Character(ch)
        self.WordCharacters.append(l)
        return l

    def restrictions(self):
        ret = []
        
        # this must get good syntax later !!!!
        if (len(self.WordCharacters)>0):
            ret.append(self.WordCharacters[0].getVar('left')-self.getVar('left'))
            ll=self.WordCharacters
            ret += for_all(ll, lambda a: ll[a].getVar('top')-self.getVar('top'))
            ret += for_all(ll, lambda a: ll[a].getVar('bottom')-self.getVar('bottom'))
            ret += between(ll, lambda a: ll[a].getVar('left')-ll[a-1].getVar('right'))
            ret.append(self.WordCharacters[len(self.WordCharacters)-1].getVar('right')-self.getVar('right'))
            
        return ret

class Line(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'height', 'width', 'right', 'bottom'] #Later this should be syntactically improved
    
        BaseRules.__init__(self)
        self.LineWords = self.childs
        
    def takeToMuch(self, ToMuch):
        self.LineWords.insert[0,ToMuch]

    def addWord(self):
        l=Word()
        self.LineWords.append(l)
        return l

    def restrictions(self):
        ret = []
        # this must get good syntax later !!!!
        ll=self.LineWords
        #print('vor', ll)
        if (len(ll)>0):
            ret.append(self.LineWords[0].getVar('left')-self.getVar('left'))
            ret += between(ll, lambda a: ll[a].getVar('left')-ll[a-1].getVar('right')-5)
            #print('ll', ll)
            ret.append(min([l.getVar('top') for l in ll])-self.getVar('top'))
            dd = [(l.getVar('bottom') -self.getVar('bottom')) for l in ll]
            ret += dd
            
        return ret

    def check_to_long(self):
        ToLong = None
        ll=self.LineWords
        if (len(ll)>0):
            if ll[len(ll)-1].getVar('right') > self.getVar('right'):
                #too long must be managed here as allowed operation
                ToLong = ll.pop()
        return ToLong

class Page(BaseRules):
    def __init__(self):
        self.priority = [] #Later this should be syntactically improved
        self.not_optimizing = ['top', 'left', 'right', 'bottom']
        BaseRules.__init__(self)
        self.PageLines = self.childs
        
    def takeToMuch(self, ToMuch):
        self.PageLines.insert[0,ToMuch]
    
    def addLine(self):
        l=Line()
        self.PageLines.append(l)
        return l
        
    def restrictions(self):
        ret = []
        # this must get good syntax later !!!!
        ll=self.PageLines
        if (len(ll)>0):
            ret.append(self.PageLines[0].getVar('top')-self.getVar('top'))
            #ret.append(self.PageLines[0].getVar('left')-self.getVar('left'))
            #ret.append(self.PageLines[0].getVar('right')-self.getVar('right'))
            ret += for_all(ll, lambda a: ll[a].getVar('left')-self.getVar('left'))
            ret += for_all(ll, lambda a: ll[a].getVar('right')-self.getVar('right'))
            ret += between(ll, lambda a: ll[a].getVar('top')-ll[a-1].getVar('bottom'))
        return ret

    def check_to_long(self):
        global actualLine
        add = None
        for l in self.PageLines:
            if add is not None:
                l.LineWords.insert(0,add)
            add = l.check_to_long()
        if add is not None:
            actualLine = self.addLine()
            actualLine.LineWords.append(add)
            add = None
        ToLong = None
        ll=self.PageLines
        if (len(ll)>0):
            if ll[len(ll)-1].getVar('bottom') > self.getVar('bottom'):
                #too long must be managed here as allowed operation
                ToLong = ll.pop()
        return ToLong


# takeToMuch can take from before, restrictions can return ToLong

testpage = Page()
testpage.setVar('left',0)
testpage.setVar('top',20)
testpage.setVar('right',400)
testpage.setVar('bottom',400)

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
    c = w.create_text(200, 300, anchor=W, font=("Times New Roman", int(25), "bold"),
            text=event.char)
    print('key pressed',event, 'bounding', w.bbox(c))
    ch=event.char
    if ch== ' ':
        testpage.check_to_long()
        actualWord = actualLine.addWord()
        for _ in range(1):
            testpage.optimize()
        w.delete("all")
        for d in testpage.get_all_self_and_childs():
            d.draw()
        
    else:
        l = actualWord.addCharacter(ch)
        (left,top,right,bottom) = w.bbox(c)
        l.setVar('left', left)
        l.setVar('width', right-left)
        l.setVar('top', top)
        l.setVar('height', bottom-top)
        for _ in range(1):
            testpage.optimize()
        w.delete("all")
        for d in testpage.get_all_self_and_childs():
            d.draw()
        testpage.full_restrictions(debug=1)
        testpage.optimize_jakobi()
        
w.bind('<Button-1>', click)
master.bind('<Key>',key)
mainloop()