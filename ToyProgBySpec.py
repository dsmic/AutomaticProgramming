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
from tokenize import tokenize
from io import BytesIO

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



class BaseRules():
    all_vars_used = {}
    def draw(self):
        #print(self, 'char nodraw', round(self.getVar('left')), round(self.getVar('top')), round(self.getVar('right')), round(self.getVar('bottom')))
        pass
    
    def __init__(self):
        #manageing variables (not used in later syntax)
        self.clean()
        self.childs = []
        print('starting',type(self).__name__)
        
        #create the properties
        for l in self.priority:
            print(l)
            s1='def gvar_'+l+'(self): return self.getVar("'+l+'")'
            s2='def svar_'+l+'(self, x): return self.setVar("'+l+'",x)' # not sure if this can be used later?!
            s3='BaseRules.' + l +' = property(gvar_'+l+',svar_'+l+')'
            print(s1)
            print(s2)
            print(s3)
            exec(s1)
            exec(s2)
            exec(s3)
            print('???',eval('self.'+l))
            
    def clean(self):
        self.TheVars = {} #contains the variables from priority
        for l in self.priority:
            self.setVar(l,0)
        
            
    def getVar(self, name):
        #global all_vars_used
        val = self.TheVars[name]
        #print('getVar val',val, type(val))
        if isinstance(val, tuple):
            obj, code = val
            #print(len(obj.childs),code)
            return eval(code,{'self': obj})
        else:
            if name in self.priority:
                self.all_vars_used[(self,name)] = 1
            return val
    
    def setVar(self, name, value):
        if name in self.TheVars and isinstance(self.TheVars[name],tuple):
            print('name is',name, self.TheVars[name], value)
            if not isinstance(value,tuple):
                return False
            if self.TheVars[name] != value:
                #raise ValueError('resetting calculation ln existing var '+str( self)+' '+str(type(self).__name__)+" "+ name + str( value)+ ' was ' +str(self.TheVars[name]))
                return False
        self.TheVars[name]=value
        return True

    def takeToMuch(self, _):
        raise ValueError('Can not take Elements from before.')
    
    def full_restrictions(self, debug = 0):
        ret = self.restrictions()
        print('r',ret)
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
        #global all_vars_used
        jakobi_list = []
        self.all_vars_used.clear()
        before = self.full_restrictions()
        all_vars_opt = [l for l in self.all_vars_used.keys()]
        if debug:
            print('vars_to_opt',all_vars_opt)
            print('before',before)
        for (obj,vv) in all_vars_opt:
                tmp = obj.getVar(vv)
                obj.setVar(vv, tmp + 1)
                after = self.full_restrictions()
                print('###',before,after)
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

    def try_set_new(self, name, thecode):
        old_set = name.rsplit('.',1)
        print('setting',old_set[0],old_set[1])
        if len(old_set)==0:
            raise ValueError('should not be possible')
        else:
            return self.try_set(old_set[0],old_set[1],thecode)
         

    def rule(self, string, child_name='self.childs'):
        def replace_names(string, child_name, i=None):
            testtokens = tokenize(BytesIO(string.encode('utf-8')).readline)
            new_string =''
            afterdot = False
            for tt in testtokens:
                print(tt)
                if tt.type == 1:
                    # here string replacement will be possible
                    ttt = tt.string
                    if afterdot:
                        new_string += ttt
                        afterdot = False
                    elif ttt == 'firstchild':
                        new_string += child_name +"[0]"
                    elif ttt == 'lastchild':
                        new_string += child_name +"[-1]"
                    elif ttt =='child' or ttt =='rightchild':
                        if i == None:
                            new_string += child_name +"[i]"
                        else:
                            new_string += child_name +"["+str(i)+"]"
                    elif ttt == 'leftchild':
                        new_string += child_name +"["+str(i-1)+"]"
                    elif len(ttt)>0 and not ttt[0].isdigit:
                        assert(False)
                        new_string += "self."+ttt
                    else:
                        new_string += "self."+ttt
                    afterdot=False
                elif tt.type !=59:
                    new_string +=tt.string
                    if tt.string == '.': afterdot=True
            print('un',new_string)
            return new_string
        
        ll=string.split(':')
        if len(ll) == 1:
            lleq = ll[0].split('=')
            if len(lleq) == 1:
                ret = '['+replace_names(ll[0],child_name)+']'
            else:
                right_side = replace_names(lleq[1],child_name)
                left_side = replace_names(lleq[0],child_name)
                print('new_set',left_side,right_side)
                ret = '['+self.try_set_new(left_side, right_side)+']'
        else:
            if ll[0] == 'for_all':
                ret = '['
                for i in range(len(eval(child_name))):
                    if i>0: ret +=','
                    lleq = ll[1].split('=')
                    if len(lleq) == 1:
                        ret += replace_names(ll[1], child_name, i)
                    else:
                        right_side = replace_names(lleq[1],child_name, i)
                        left_side = replace_names(lleq[0],child_name, i)
                        print('new_set',left_side,right_side)
                        ret += self.try_set_new(left_side, right_side)
                        if ret[-1]==',': ret = ret[:-1]
                ret +=']'
            elif ll[0] =='between':
                print('inbetween',ll[1],len(eval(child_name)))
                ret = '['
                for i in range(1,len(eval(child_name))):
                    if i>1: ret +=','
                    lleq = ll[1].split('=')
                    if len(lleq) == 1:
                        ret += replace_names(ll[1], child_name, i)
                    else:
                        right_side = replace_names(lleq[1],child_name, i)
                        left_side = replace_names(lleq[0],child_name, i)
                        print('new_set',left_side,right_side)
                        ret += self.try_set_new(left_side, right_side)
                        if ret[-1]==',': ret = ret[:-1]
                        #raise ValueError('not yet implemented')
                    print('ret',i,ret)
                ret +=']'
            elif ll[0] =='min_all':
                ret = 'min_all('+child_name +', lambda i: '
                ret += replace_names(ll[1], child_name)
                ret +=')'
                print('min_all', ret)
                #raise ValueError('not implemented')
            print(ll[0])
            
        print('##',ret)
        return eval(ret,dict(self=self, for_all=for_all, min_all=min_all, between=between))
             

    def rule_old(self, string, child_name='self.childs'):
        string=string.replace(" ", "")
        ret=''
        ll=string.split(':')
        if len(ll) == 1:
            #ll[0] = ll[0].split('=')[0] # allows =0 at the end, just to keep syntax like equation solver
            ret+='['
        
            # we start here to test if it can be done by a reference
            # ll2[0] is what has to be set to the rest of ll2 as sum to be compiled
            # ret must be an empty list than
            ret = '['
            thecode = ''
            testtokens = tokenize(BytesIO(ll[0].encode('utf-8')).readline)
            print(ll[0])
            new_string =''
            for tt in testtokens:
                if tt.type != 59:
                    # here string replacement will be possible
                    new_string += tt.string
                print(tt)
            print('un',new_string)
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
                        thecode+=child_name +"[0].getVar('"+ll4[1]+"')"
                    elif  ll4[0]=='lastchild':
                        thecode+=child_name +"[-1].getVar('"+ll4[1]+"')"
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
                    ret += self.try_set(child_name +'[0]', ll4[1], thecode)
                elif  ll4[0]=='lastchild':
                    ret += self.try_set(child_name +'[-1]', ll4[1], thecode)
                else:
                    raise ValueError('not firstchild or lastchild')
            ret +=']'
        else:
            ll[1] = ll[1].split('=')[0] # allows =0 at the end, just to keep syntax like equation solver
            if ll[0] == 'for_all':
                ret = '['
                for i in range(len(eval(child_name))):
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
                            thecode += child_name +"["+str(i)+"].getVar('"+ll4[1]+"')"
                        thecode+='+'
                    thecode=thecode[:-1]
                    ll4 = firstis.split('.')
                    if len(ll4) == 2:
                        assert(ll4[0]=='child')
                        tmp = self.try_set(child_name +'['+str(i)+']', ll4[1], thecode)
                        if len(tmp)>0:
                            ret +=  child_name +"["+str(i)+"].getVar('"+ll4[1]+"') - ("+thecode+')' # a comma seems to be missing here?!
                            #raise ValueError('tryset failed '+thecode)
                    else:
                        raise ValueError('for all with first child parameter forced')
                ret +=']'
                    
            elif ll[0] == 'min_all':
                #print('minall',ret)
                ret +='min_all('+child_name +', lambda a: '
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
                        ret += child_name +"[a].getVar('"+ll4[1]+"')"
                    ret+='-'
                ret=ret[:-1]+')'
                #print('min_all____',ret)
            elif ll[0] == 'between':
                ret = '['
                for i in range(1,len(eval(child_name))):
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
                            thecode += child_name +"["+str(i-1)+"].getVar('"+ll4[1]+"')"
                        thecode+='+'
                    thecode=thecode[:-1]
                    ll4 = firstis.split('.')
                    if len(ll4) == 2:
                        assert(ll4[0]=='rightchild')
                        tmp = self.try_set(child_name +'['+str(i)+']', ll4[1], thecode)
                        if len(tmp)>0:
                            ret +=  child_name +"["+str(i)+"].getVar('"+ll4[1]+"') - ("+thecode+')'
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
        return self.rule('bottom=top+height') + self.rule('right-left-width')
    def gvar_height(self): return self.getVar('height')
    def gvar_width(self): return self.getVar('width')
    height = property(gvar_height)
    width = property(gvar_width)
    
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
            ret += self.rule('firstchild.left=left')
            ret += self.rule('for_all: child.top=top')
            ret += self.rule('for_all: child.bottom-bottom')
            ret += self.rule('between: rightchild.left=leftchild.right')
            ret += self.rule('right-lastchild.right')
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
            ret += self.rule('firstchild.left - left')
            ret += self.rule('between: rightchild.left-leftchild.right-5')
            ret += self.rule('min_all: child.top - top')
            ret += self.rule('for_all: child.bottom-bottom')
            
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
            ret += self.rule('firstchild.top - top ')
            ret += self.rule('for_all: child.left-left')
            ret += self.rule('for_all: child.right-right')
            ret += self.rule('between: rightchild.top - leftchild.bottom')
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



print(testpage.bottom)