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
    #this is not really min, it must be possible to solve equation system with gradient
    ret=None
    sum_neg=0
    for i in range(0,len(ll)):
        t=compare(i)
        #print('in min',i,t)
        if ret == None or t<ret:
            ret = t
        if t<0:
            sum_neg+=t
    if sum_neg < 0:
        return [sum_neg]
    else:
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
        self.clean()
        self.childs = []
        
    def clean(self):
        self.TheVars = {} #contains the variables from priority
        for l in self.priority:
            self.setVar(l,0)
        
            
    def getVar(self, name):
        global all_vars_used
        val = self.TheVars[name]
        #print('getVar val',val, type(val))
        if isinstance(val, tuple):
            obj, code = val
            #print(len(obj.childs),code)
            return eval(code,{'self': obj})
        else:
            if name in self.priority:
                all_vars_used[(self,name)] = 1
            return val
    
    def setVar(self, name, value):
        if name in self.TheVars and isinstance(self.TheVars[name],tuple):
            if self.TheVars[name] != value:
                #raise ValueError('resetting calculation ln existing var '+str( self)+' '+str(type(self).__name__)+" "+ name + str( value)+ ' was ' +str(self.TheVars[name]))
                return False
        self.TheVars[name]=value
        return True

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
        
    def optimize(self, debug=False):
        global all_vars_used
        jakobi_list = []
        all_vars_used.clear()
        before = self.full_restrictions()
        all_vars_opt = [l for l in all_vars_used.keys()]
        if debug:
            print('vars_to_opt',all_vars_opt)
            print('before',before)
        for (obj,vv) in all_vars_opt:
                tmp = obj.getVar(vv)
                obj.setVar(vv, tmp + 1)
                after = self.full_restrictions()
                #print('###',before,after)
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
                print('-opt-',i,d)
            i+=1
        
    def optimize_nonjakobi(self):
        #old version, does not handle references
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

    def try_set(self, where, name, thecode):
        if name in eval(where).priority:
            if eval(where).setVar(name,(self,thecode)):
                return ''
        #print('not setable',type(self).__name__, where, name, thecode,  eval(where).priority, name in eval(where).priority)
        return where+".getVar('"+name+"')-("+thecode+")"


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
                if len(ll4) == 1 and not ll4[0][0].isdigit():
                    ret += self.try_set('self', ll4[0], thecode)
                    #print('thecode',ret,thecode,self.getVar('bottom'),self.getVar('top'))
                else:
                    if  ll4[0]=='firstchild':
                        ret += self.try_set('self.childs[0]', ll4[1], thecode)
                        # if ll4[1] in self.childs[0].priority:
                        #     self.childs[0].setVar(ll4[1],(self,thecode))
                        # else:
                        #     ret += "self.childs[0].getVar('"+ll4[1]+"')-("+thecode+")"
                    elif  ll4[0]=='lastchild':
                        ret += self.try_set('self.childs[-1]', ll4[1], thecode)
                        # if ll4[1] in self.childs[-1].priority:
                        #     self.childs[-1].setVar(ll4[1],(self,thecode))
                        # else:
                        #     ret += "self.childs[-1].getVar('"+ll4[1]+"')-("+thecode+")"
                    else:
                        raise ValueError('not firstchild or lastchild')
                ret +=']'
        else:
            ll[1] = ll[1].split('=')[0] # allows =0 at the end, just to keep syntax like equation solver
            if ll[0] == 'for_all':
                if no_references:
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
                else:
                    ret = '['
                    for i in range(len(self.childs)):
                        ll2=ll[1].split('-')
                        firstis = ll2.pop(0)
                        thecode = ''
                        for ll3 in ll2:
                            ll4 = ll3.split('.')
                            #print(ll4)
                            if len(ll4) == 1:
                                if ll4[0][0].isdigit():
                                    thecode +=ll4[0]
                                else:
                                    thecode += "self.getVar('"+ll4[0]+"')"
                            else:
                                assert(ll4[0]=='child')
                                thecode += "self.childs["+str(i)+"].getVar('"+ll4[1]+"')"
                            thecode+='+'
                        thecode=thecode[:-1]
                        ll4 = firstis.split('.')
                        if len(ll4) == 2:
                            assert(ll4[0]=='child')
                            tmp = self.try_set('self.childs['+str(i)+']', ll4[1], thecode)
                            if len(tmp)>0:
                                ret +=  "self.childs["+str(i)+"].getVar('"+ll4[1]+"') - ("+thecode+')'
                                #raise ValueError('tryset failed '+thecode)
                        else:
                            raise ValueError('for all with first child parameter forced')
                    ret +=']'
                    
            elif ll[0] == 'min_all':
                #print('minall',ret)
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
                #print('min_all____',ret)
            elif ll[0] == 'between':
                if no_references:
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
                else:
                    ret = '['
                    for i in range(1,len(self.childs)):
                        ll2=ll[1].split('-')
                        firstis = ll2.pop(0)
                        thecode = ''
                        for ll3 in ll2:
                            ll4 = ll3.split('.')
                            #print(ll4)
                            if len(ll4) == 1:
                                if ll4[0][0].isdigit():
                                    thecode +=ll4[0]
                                else:
                                    thecode += "self.getVar('"+ll4[0]+"')"
                            else:
                                assert(ll4[0]=='leftchild')
                                thecode += "self.childs["+str(i-1)+"].getVar('"+ll4[1]+"')"
                            thecode+='+'
                        thecode=thecode[:-1]
                        ll4 = firstis.split('.')
                        if len(ll4) == 2:
                            assert(ll4[0]=='rightchild')
                            tmp = self.try_set('self.childs['+str(i)+']', ll4[1], thecode)
                            if len(tmp)>0:
                                ret +=  "self.childs["+str(i)+"].getVar('"+ll4[1]+"') - ("+thecode+')'
                                #raise ValueError('tryset failed '+thecode)
                        else:
                            raise ValueError('for all with first child parameter forced')
                    ret += ']'
                    
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
        #print('rule',self.rule('bottom-top-height=0'))
        return self.rule('bottom-top-height=0') + self.rule('right-left-width=0')
    
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
            ret += self.rule('firstchild.left-left=0')
            ret += self.rule('for_all: child.top-top=0')
            ret += self.rule('for_all: child.bottom-bottom=0')
            ret += self.rule('between: rightchild.left-leftchild.right=0')
            ret += self.rule('right-lastchild.right=0')
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
            ret += self.rule('firstchild.left - left=0')
            ret += self.rule('between: rightchild.left-leftchild.right-5=0')
            ret += self.rule('min_all: child.top - top=0')
            ret += self.rule('for_all: child.bottom-bottom=0')
            
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
            ret += self.rule('firstchild.top - top =0')
            ret += self.rule('for_all: child.left-left=0')
            ret += self.rule('for_all: child.right-right=0')
            ret += self.rule('between: rightchild.top - leftchild.bottom=0')
        return ret

    def check_to_long(self):
        global actualLine
        add = None
        for l in self.childs:
            if add is not None:
                l.childs.insert(0,add)
            add = l.check_to_long()
        if add is not None:
            add.clean()
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

testpage.setVar('left',0)
testpage.setVar('top',200)
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

def printinfos():
    print('Page')
    for k,v in testpage.TheVars.items():
        print('  ',k,v, testpage.getVar(k))
    for l in testpage.childs:
        print('Line')
        for k,v in l.TheVars.items():
            print('     ',k,v, l.getVar(k))
        for w in l.childs:
            print('Word')
            for k,v in w.TheVars.items():
                print('        ',k,v,w.getVar(k))
            for c in w.childs:
                print('Char')
                for k,v in c.TheVars.items():
                    print('            ',k,v, c.getVar(k))
                    
                    
            

def key(event):
    global actualWord
    c = w.create_text(200, 300, anchor=W, font=("Times New Roman", int(25), "bold"),
            text=event.char)
    print('key pressed',event, 'bounding', w.bbox(c))
    ch=event.char
    if ch== ' ':
        testpage.check_to_long()
        actualWord = actualLine.addWord()
        for pos in range(5):
            print('-----',pos,'---')
            testpage.optimize()
        w.delete("all")
        for d in testpage.get_all_self_and_childs():
            d.draw()
        full=testpage.full_restrictions(debug=0)
        #testpage.optimize()
        print('full',full)
        #printinfos()
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
#        for lines in testpage.childs:
#            print(lines.TheVars)
        
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

