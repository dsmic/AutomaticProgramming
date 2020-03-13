#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:53:25 2020

@author: detlef
"""


# pylint: disable=C0301, C0103, C0116, C0321, C0115, R0914, R0912, R0915, R1705, R1720, W0122, W0603, W0123


from tkinter import Tk, Canvas, mainloop, W
import operator
from tokenize import tokenize
from io import BytesIO
import numpy as np

def getVariable(name):
    return name.getVar

# takes a lambda function at the moment to compare two elements
    # e.g. between(ll,lambda a:ll[a].getVar('left')-ll[a-1].getVar('right'))
def min_all(ll, compare):
    #this is not really min, it must be possible to solve equation system with gradient
    ret = None
    sum_neg = 0
    for i in range(0, len(ll)):
        t = compare(i)
        if ret is None or t < ret:
            ret = t
        if t < 0:
            sum_neg += t
    if sum_neg < 0:
        return [sum_neg]
    else:
        return [ret]

def for_all(ll, compare):
    ret = []
    for i in range(0, len(ll)):
        ret.append(compare(i))
    return ret

def between(ll, compare):
    ret = []
    for i in range(1, len(ll)):
        ret.append(compare(i))
    return ret



class BaseRules():
    priority = None # has to be overwritten by child
    all_vars_used = {}
    def draw(self):
        #print(self, 'char nodraw', round(self.getVar('left')), round(self.getVar('top')), round(self.getVar('right')), round(self.getVar('bottom')))
        pass

    def __init__(self):
        #manageing variables (not used in later syntax)
        self.clean()
        self.childs = []

        #create the properties
        for l in self.priority:
            #print(l)
            s1 = 'def gvar_'+l+'(self): return self.getVar("'+l+'")'
            s2 = 'def svar_'+l+'(self, x): return self.setVar("'+l+'",x)' # not sure if this can be used later?!
            s3 = 'BaseRules.' + l +' = property(gvar_'+l+',svar_'+l+')'
            exec(s1)
            exec(s2)
            exec(s3)

    def clean(self):
        self.TheVars = {} #contains the variables from priority
        for l in self.priority:
            self.setVar(l, 0)


    def getVar(self, name):
        val = self.TheVars[name]
        if isinstance(val, tuple):
            obj, code = val
            return eval(code, {'self': obj})
        else:
            if name in self.priority:
                self.all_vars_used[(self, name)] = 1
            return val

    def setVar(self, name, value):
        if name in self.TheVars and isinstance(self.TheVars[name], tuple):
            #print('name is',name, self.TheVars[name], value)
            if not isinstance(value, tuple):
                return False
            if self.TheVars[name] != value:
                return False
        self.TheVars[name] = value
        return True

    def full_restrictions(self, debug=0):
        ret = self.restrictions()
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
        jakobi_list = []
        self.all_vars_used.clear()
        before = self.full_restrictions()
        all_vars_opt = self.all_vars_used.keys()
        if debug:
            print('vars_to_opt', all_vars_opt)
            print('before', before)
        for (obj, vv) in all_vars_opt:
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
        i = 0
        for (obj, vv) in all_vars_opt:
            obj.setVar(vv, obj.getVar(vv) - delta[i])
            i += 1
        check = self.full_restrictions()
        i = 0
        print('len of optimizing', len(check))
        for d in check:
            if abs(d) > 3:
                print('-opt-', i, d)
            i += 1

    def try_set(self, where, name, thecode):
        if name in eval(where).priority:
            if eval(where).setVar(name, (self, thecode)):
                return ''
        return where+".getVar('"+name+"')-("+thecode+")"

    def try_set_new(self, name, thecode):
        old_set = name.rsplit('.', 1)
        if len(old_set) == 0:
            raise ValueError('should not be possible')
        else:
            return self.try_set(old_set[0], old_set[1], thecode)


    def rule(self, rulestring, child_name='self.childs'):
        """
        Parameters
        ----------
        rulestring : TYPE
            becomes zero for the correct values, if not in equation
            if equation is used
            e.g.:  top = bottom - height
            it tries to create a reference in the top variable
            supported expressions are:
                for_all: reference to the child
                between: reference to leftchild and rightchild
                min_all: use child, but can not create references
        child_name : TYPE, optional
            DESCRIPTION. The default is 'self.childs'. Defines what child, firstchild and rightchild
            in the rulestring is beeing referenced to.

        Returns
        -------
        TYPE
            List of elements for the optimizer. References are created as side effect

        """
        def replace_names(string, child_name, i=None):
            testtokens = tokenize(BytesIO(string.encode('utf-8')).readline)
            new_string = ' '
            afterdot = False
            for tt in testtokens:
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
                    elif ttt in ('child', 'rightchild'):
                        if i is None:
                            new_string += child_name +"[i]"
                        else:
                            new_string += child_name +"["+str(i)+"]"
                    elif ttt == 'leftchild':
                        new_string += child_name +"["+str(i-1)+"]"
                    else:
                        new_string += "self."+ttt
                    afterdot = False
                elif tt.type != 59:
                    new_string += tt.string
                    if tt.string == '.': afterdot = True
            return new_string

        ll = rulestring.split(':')
        if len(ll) == 1:
            lleq = ll[0].split('=')
            if len(lleq) == 1:
                ret = '['+replace_names(ll[0], child_name)+']'
            else:
                right_side = replace_names(lleq[1], child_name)
                left_side = replace_names(lleq[0], child_name)
                ret = '['+self.try_set_new(left_side, right_side)+']'
        else:
            if ll[0] == 'for_all':
                ret = '['
                for i in range(len(eval(child_name))):
                    if i > 0: ret += ','
                    lleq = ll[1].split('=')
                    if len(lleq) == 1:
                        ret += replace_names(ll[1], child_name, i)
                    else:
                        right_side = replace_names(lleq[1], child_name, i)
                        left_side = replace_names(lleq[0], child_name, i)
                        ret += self.try_set_new(left_side, right_side)
                        if ret[-1] == ',': ret = ret[:-1]
                ret += ']'
            elif ll[0] == 'between':
                ret = '['
                for i in range(1, len(eval(child_name))):
                    if i > 1: ret += ','
                    lleq = ll[1].split('=')
                    if len(lleq) == 1:
                        ret += replace_names(ll[1], child_name, i)
                    else:
                        right_side = replace_names(lleq[1], child_name, i)
                        left_side = replace_names(lleq[0], child_name, i)
                        ret += self.try_set_new(left_side, right_side)
                        if ret[-1] == ',': ret = ret[:-1]
                ret += ']'
            elif ll[0] == 'min_all':
                # here references are not possible
                ret = 'min_all('+child_name +', lambda i: '
                ret += replace_names(ll[1], child_name)
                ret += ')'

        return eval(ret, dict(self=self, for_all=for_all, min_all=min_all, between=between))

    def restrictions(self):
        """
        are created from rules by adding the return value lists
        """
        # pylint: disable=R0201, W0101
        raise ValueError('This has to be overwritten by the child class')
        return []

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
        return self.rule('bottom=top+height') + self.rule('right=left+width')
    def gvar_height(self): return self.getVar('height')
    def gvar_width(self): return self.getVar('width')
    height = property(gvar_height)
    width = property(gvar_width)

class Word(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved

        BaseRules.__init__(self)

    def addCharacter(self, ch):
        l = Character(ch)
        self.childs.append(l)
        return l

    def restrictions(self):
        ret = []
        if len(self.childs) > 0:
            ret += self.rule('firstchild.left=left')
            ret += self.rule('for_all: child.top=top')
            ret += self.rule('for_all: child.bottom=bottom')
            ret += self.rule('between: rightchild.left=leftchild.right')
            ret += self.rule('right=lastchild.right')
        return ret

class Line(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved

        BaseRules.__init__(self)

    def addWord(self):
        l = Word()
        self.childs.append(l)
        return l

    def restrictions(self):
        ret = []
        ll = self.childs
        if len(ll) > 0:
            ret += self.rule('firstchild.left = left')
            ret += self.rule('between: rightchild.left=leftchild.right+5')
            ret += self.rule('min_all: child.top - top')
            ret += self.rule('for_all: child.bottom=bottom')

        return ret

    def check_to_long(self):
        ToLong = None
        ll = self.childs
        if len(ll) > 0:
            if ll[len(ll)-1].getVar('right') > self.getVar('right'):
                ToLong = ll.pop()
        return ToLong



class Page(BaseRules):
    def __init__(self):
        self.priority = [] #Later this should be syntactically improved
        BaseRules.__init__(self)

    def addLine(self):
        l = Line()
        self.childs.append(l)
        return l

    def restrictions(self):
        ret = []
        # this must get good syntax later !!!!
        ll = self.childs
        if len(ll) > 0:
            ret += self.rule('firstchild.top = top ')
            ret += self.rule('for_all: child.left=left')
            ret += self.rule('for_all: child.right=right')
            ret += self.rule('between: rightchild.top = leftchild.bottom')
        return ret

    def check_to_long(self):
        global actualLine
        add = None
        for l in self.childs:
            if add is not None:
                l.childs.insert(0, add)
            add = l.check_to_long()
        if add is not None:
            add.clean()
            actualLine = self.addLine()
            actualLine.childs.append(add)
            add = None
        ToLong = None
        ll = self.childs
        if len(ll) > 0:
            if ll[len(ll)-1].getVar('bottom') > self.getVar('bottom'):
                ToLong = ll.pop()
        return ToLong


testpage = Page()

testpage.setVar('left', 0)
testpage.setVar('top', 200)
testpage.setVar('right', 400)
testpage.setVar('bottom', 400)

actualLine = testpage.addLine()
actualWord = actualLine.addWord()

# Toy Manager
master = Tk()
canvas_width = 800
canvas_height = 400
w = Canvas(master, width=canvas_width, height=canvas_height)
w.pack()

def click(event):
    print('button clicked', event)

def printinfos():
    print('Page')
    for k, v in testpage.TheVars.items():
        print('  ', k, v, testpage.getVar(k))
    for l in testpage.childs:
        print('Line')
        for k, v in l.TheVars.items():
            print('     ', k, v, l.getVar(k))
        for ww in l.childs:
            print('Word')
            for k, v in ww.TheVars.items():
                print('        ', k, v, ww.getVar(k))
            for c in ww.childs:
                print('Char')
                for k, v in c.TheVars.items():
                    print('            ', k, v, c.getVar(k))




def key(event):
    global actualWord
    c = w.create_text(200, 300, anchor=W, font=("Times New Roman", int(25), "bold"),
                      text=event.char)
    print('key pressed', event, 'bounding', w.bbox(c))
    ch = event.char
    if ch == ' ':
        testpage.check_to_long()
        actualWord = actualLine.addWord()
        for pos in range(5):
            print('-----', pos, '---')
            testpage.optimize()
        w.delete("all")
        for d in testpage.get_all_self_and_childs():
            d.draw()
        full = testpage.full_restrictions(debug=0)
        #testpage.optimize()
        print('full', full)
        #printinfos()
    else:
        l = actualWord.addCharacter(ch)
        (left, top, right, bottom) = w.bbox(c)
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
master.bind('<Key>', key)
mainloop()
