#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:53:25 2020

@author: detlef
"""

from tkinter import Tk, Canvas, mainloop, W
import numpy as np
import operator
import types

no_references = False
                
def getVariable(name):
    return name.getVar            

# takes a lambda function at the moment to compare two elements
    # e.g. between(ll,lambda a:ll[a].getVar('left')-ll[a-1].getVar('right'))
def min_all(ll, compare):
    ret=None
    for i in range(0,len(ll)):
        t=compare(i)
        if ret == None or t<ret:
            ret = t
    return [ret]

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

all_vars_used = {}

class BaseRules():
    def draw(self):
        #print(self, 'char nodraw', round(self.getVar('left')), round(self.getVar('top')), round(self.getVar('right')), round(self.getVar('bottom')))
        pass
    
    def __init__(self):
        #manageing variables (not used in later syntax)
        self.TheVars = {} #contains the variables from priority
        for l in self.priority:
            self.setVar(l,0)
        self.childs = []
            
    def getVar(self, name):
        global all_vars_used
        val = self.TheVars[name]
        #print('getVar val',val, type(val))
        if isinstance(val, types.CodeType):
            return eval(val)
        else:
            if name in self.priority:
                all_vars_used[(self,name)] = 1
            return val
    
    def setVar(self, name, value):
        if name in self.TheVars and isinstance(self.TheVars[name],types.CodeType):
            raise ValueError('resetting calculation ln existing var')
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
        
    def optimize(self):
        global all_vars_used
        jakobi_list = []
        all_vars_used.clear()
        before = self.full_restrictions()
        all_vars_opt = [l for l in all_vars_used.keys()]
        for (obj,vv) in all_vars_opt:
        
                tmp = obj.getVar(vv)
                obj.setVar(vv, tmp + 1)
                after = self.full_restrictions()
                diff = list(map(operator.sub, after, before))
                obj.setVar(vv, tmp)
                jakobi_list.append(diff)
        f_x = np.array(before)
        JK = np.array(jakobi_list)
        JK_t = JK.transpose()
        JK_t_i = np.linalg.pinv(JK_t, rcond=0.001)
        delta = np.dot(JK_t_i, f_x)
        i=0
        for (obj,vv) in all_vars_opt:
                obj.setVar(vv, obj.getVar(vv) - delta[i])
                i += 1
        check = self.full_restrictions()
        i=0
        for d in check:
            if abs(d)>3:
                print(i,d)
            i+=1
        
    def optimize_nonjakobi(self):
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
                correct -= 1        
                obj.setVar(vv, obj.getVar(vv)+correct)

    def rule(self, string):
        string=string.replace(" ", "")
        ret=''
        ll=string.split(':')
        if len(ll) == 1:
            ll[0] = ll[0].split('=')[0] # allows =0 at the end, just to keep syntax like equation solver
            ret+='['
        
            # we start here to test if it can be done by a reference
            # ll2[0] is what has to be set to the rest of ll2 as sum to be compiled
            # ret must be an empty list than
            if no_references:
                ll2=ll[0].split('-')
                for ll3 in ll2:
                    ll4 = ll3.split('.')
                    if len(ll4) == 1:
                            if ll4[0][0].isdigit():
                                ret +=ll4[0]
                            else:
                                ret += "self.getVar('"+ll4[0]+"')"
                    else:
                        if  ll4[0]=='firstchild':
                            ret+="self.childs[0].getVar('"+ll4[1]+"')"
                        elif  ll4[0]=='lastchild':
                            ret+="self.childs[-1].getVar('"+ll4[1]+"')"
                        else:
                            raise ValueError('not firstchild or lastchild')
                    ret+='-'
                ret=ret[:-1]+']'
            else:
                ret = '['
                thecode = ''
                ll2=ll[0].split('-')
                firstis = ll2.pop(0)
                for ll3 in ll2:
                    ll4 = ll3.split('.')
                    if len(ll4) == 1:
                            if ll4[0][0].isdigit():
                                thecode +=ll4[0]
                            else:
                                thecode += "self.getVar('"+ll4[0]+"')"
                    else:
                        if  ll4[0]=='firstchild':
                            thecode+="self.childs[0].getVar('"+ll4[1]+"')"
                        elif  ll4[0]=='lastchild':
                            thecode+="self.childs[-1].getVar('"+ll4[1]+"')"
                        else:
                            raise ValueError('not firstchild or lastchild')
                    thecode+='+'
                thecode = thecode[:-1]
                #print("setting",thecode)
                ll4 = firstis.split('.')
                ref_code = compile(thecode,'<stdin>','eval')
                if len(ll4) == 1 and not ll4[0][0].isdigit():
                    if ll4 in self.priority:
                        self.setVar(ll4[0],ref_code)
                    else:
                        ret += "self.getVar('"+ll4[0]+"')-("+thecode+")"
                        #print('nor setable',ret)
                else:
                    raise ValueError('first in simple rule not availible')
                ret +=']'
        else:
            ll[1] = ll[1].split('=')[0] # allows =0 at the end, just to keep syntax like equation solver
            if ll[0] == 'for_all':
                ret +='for_all(self.childs, lambda a: '
                ll2=ll[1].split('-')
                for ll3 in ll2:
                    ll4 = ll3.split('.')
                    #print(ll4)
                    if len(ll4) == 1:
                        if ll4[0][0].isdigit():
                            ret +=ll4[0]
                        else:
                            ret += "self.getVar('"+ll4[0]+"')"
                    else:
                        assert(ll4[0]=='child')
                        ret += "self.childs[a].getVar('"+ll4[1]+"')"
                    ret+='-'
                ret=ret[:-1]+')'
            elif ll[0] == 'min_all':
                ret +='min_all(self.childs, lambda a: '
                ll2=ll[1].split('-')
                for ll3 in ll2:
                    ll4 = ll3.split('.')
                    #print(ll4)
                    if len(ll4) == 1:
                        if ll4[0][0].isdigit():
                            ret +=ll4[0]
                        else:
                            ret += "self.getVar('"+ll4[0]+"')"
                    else:
                        assert(ll4[0]=='child')
                        ret += "self.childs[a].getVar('"+ll4[1]+"')"
                    ret+='-'
                ret=ret[:-1]+')'
            elif ll[0] == 'between':
                ret +='between(self.childs, lambda a: '
                ll2=ll[1].split('-')
                for ll3 in ll2:
                    ll4 = ll3.split('.')
                    #print(ll4)
                    if len(ll4) == 1:
                        if ll4[0][0].isdigit():
                            ret +=ll4[0]
                        else:
                            ret += "self.getVar('"+ll4[0]+"')"
                    else:
                        if ll4[0]=='child':
                            ret += "self.childs[a].getVar('"+ll4[1]+"')"
                        elif ll4[0]=='leftchild':
                            ret += "self.childs[a-1].getVar('"+ll4[1]+"')"
                        elif ll4[0]=='rightchild':
                            ret += "self.childs[a].getVar('"+ll4[1]+"')"
                        else:
                            raise ValueError('not child, leftchild or rightchild')
                    ret+='-'
                ret=ret[:-1]+')'
            else: raise ValueError('rule wrong',string)
        #print('rrr',ret)
        return eval(ret,dict(self=self, for_all=for_all, min_all=min_all, between=between))


class Character(BaseRules):
    def draw(self):
            #print(self, 'char draw', self.TheCharacter, round(self.getVar('left')), round(self.getVar('top')), round(self.getVar('right')), round(self.getVar('bottom')))
            w.create_text(self.getVar('left'), self.getVar('top'), anchor=W, font=("Times New Roman", int(25), "bold"),
            text=self.TheCharacter)

    def __init__(self, ch):
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved
        
        #fixed content not changed by other variables
        self.TheCharacter = ch
        BaseRules.__init__(self)
        
    def restrictions(self):
        # becomes zero for the correct values, uses priority to determine which to optimize
        # planed syntax would be:
        # top-bottom = height
        # right-left = width
        return self.rule('bottom-top-height=0') + self.rule('right-left-width=0')
        #return [self.getVar('bottom')-self.getVar('top')-self.getVar('height'), 
        #        self.getVar('right')-self.getVar('left')-self.getVar('width')]
    
class Word(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved
    
        BaseRules.__init__(self)
        
    def addCharacter(self, ch):
        l=Character(ch)
        self.childs.append(l)
        return l

    def restrictions(self):
        ret = []
        if (len(self.childs)>0):
            ret += self.rule('left-firstchild.left=0')
            #ret.append(self.childs[0].getVar('left')-self.getVar('left'))
            #ll=self.childs
            ret += self.rule('for_all: child.top-top=0')
            #ret += for_all(ll, lambda a: ll[a].getVar('top')-self.getVar('top'))
            ret += self.rule('for_all: child.bottom-bottom=0')
            #ret += for_all(ll, lambda a: ll[a].getVar('bottom')-self.getVar('bottom'))
            ret += self.rule('between: rightchild.left-leftchild.right=0')
            #ret += between(ll, lambda a: ll[a].getVar('left')-ll[a-1].getVar('right'))
            ret += self.rule('right-lastchild.right=0')
            #ret.append(self.childs[len(self.childs)-1].getVar('right')-self.getVar('right'))
            #
        return ret
    
class Line(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved
    
        BaseRules.__init__(self)
        
    def takeToMuch(self, ToMuch):
        self.childs.insert[0,ToMuch]

    def addWord(self):
        l=Word()
        self.childs.append(l)
        return l

    def restrictions(self):
        ret = []
        ll=self.childs
        if (len(ll)>0):
            ret += self.rule('left - firstchild.left=0')
            #ret.append(self.childs[0].getVar('left')-self.getVar('left'))
            ret += self.rule('between: rightchild.left-leftchild.right-5=0')
            #ret += between(ll, lambda a: ll[a].getVar('left')-ll[a-1].getVar('right')-5)
            #print('ll', ll)
            #ret.append(min([l.getVar('top') for l in ll])-self.getVar('top')) #transform to min_for_all function ????
            ret += self.rule('min_all: child.top - top=0')
            #ret += min_all(ll, lambda a:ll[a].getVar('top') - self.getVar('top'))
            #dd = [(l.getVar('bottom') -self.getVar('bottom')) for l in ll]
            #dd = for_all(ll, lambda a: ll[a].getVar('bottom') -self.getVar('bottom'))
            ret += self.rule('for_all: child.bottom-bottom=0')
            #ret += for_all(ll, lambda a: ll[a].getVar('bottom') -self.getVar('bottom'))
            
        return ret

    def check_to_long(self):
        ToLong = None
        ll=self.childs
        if (len(ll)>0):
            if ll[len(ll)-1].getVar('right') > self.getVar('right'):
                ToLong = ll.pop()
        return ToLong



class Page(BaseRules):
    def __init__(self):
        self.priority = [] #Later this should be syntactically improved
        BaseRules.__init__(self)
        
    def takeToMuch(self, ToMuch):
        self.childs.insert[0,ToMuch]
    
    def addLine(self):
        l=Line()
        self.childs.append(l)
        return l
        
    def restrictions(self):
        ret = []
        # this must get good syntax later !!!!
        ll=self.childs
        if (len(ll)>0):
            ret += self.rule('top - firstchild.top =0')
            #ret.append(self.childs[0].getVar('top')-self.getVar('top'))
            ret += self.rule('for_all: child.left-left=0')
            #ret += for_all(ll, lambda a: ll[a].getVar('left')-self.getVar('left'))
            ret += self.rule('for_all: child.right-right=0')
            #ret += for_all(ll, lambda a: ll[a].getVar('right')-self.getVar('right'))
            ret += self.rule('between: rightchild.top - leftchild.bottom=0')
            #ret += between(ll, lambda a: ll[a].getVar('top')-ll[a-1].getVar('bottom'))
        return ret

    def check_to_long(self):
        global actualLine
        add = None
        for l in self.childs:
            if add is not None:
                l.childs.insert(0,add)
            add = l.check_to_long()
        if add is not None:
            actualLine = self.addLine()
            actualLine.childs.append(add)
            add = None
        ToLong = None
        ll=self.childs
        if (len(ll)>0):
            if ll[len(ll)-1].getVar('bottom') > self.getVar('bottom'):
                ToLong = ll.pop()
        return ToLong


testpage = Page()

# this may not be possilbe to set this variables. Set variables are not in priority,
# this may be used for rule creation?!

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
        testpage.full_restrictions(debug=0)
        testpage.optimize()
        
w.bind('<Button-1>', click)
master.bind('<Key>',key)
mainloop()

# some tests
x=5
class test():
    def cc(self,name):
        self.c = compile(name,'<stdin>', 'eval')
        print(self.c,type(self.c),isinstance(self.c, types.CodeType), eval(self.c))
        
    def get(self, name):
        self.x = 42
        print(eval(self.c))
        
f=test()        
f.cc("x*3")

x=66
f.get('')



