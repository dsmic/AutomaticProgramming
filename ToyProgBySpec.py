#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:53:25 2020

@author: detlef
"""

# pylint: disable=C0301, C0103, C0116, C0321, C0115, R0914, R0912, R0915, R1705, R1720, W0122, W0603, W0123, R1702

from tkinter import Tk, Canvas, mainloop, NW
import operator
from tokenize import tokenize
from io import BytesIO
import numpy as np
import random

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
        return [sum_neg*1.1]
    else:
        return [ret*1.1]

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

    def add_property(self, l):
        classname = self.__class__.__name__
        s1 = 'def gvar_'+l+'(self): return self.getVar("'+l+'")'
        s2 = 'def svar_'+l+'(self, x): return self.setVar("'+l+'",x)' # not sure if this can be used later?!
        s3 = classname+'.' + l +' = property(gvar_'+l+',svar_'+l+')'
        exec(s1)
        exec(s2)
        exec(s3)

    def __init__(self):
        #manageing variables (not used in later syntax)
        self.TheVars = {} #contains the variables from priority
        self.clean()
        self.childs = []

        #create the properties
        for l in self.priority:
            self.add_property(l)

    def clean(self):
        for l in self.priority:
            self.TheVars[l] = 0

    def clean_down(self):
        for l in self.priority:
            if isinstance(self.TheVars[l],tuple):
                self.TheVars[l] = self.getVar(l)
        for l in self.childs:
            l.clean_down()

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
        #print('vvv',len(self.all_vars_used), self.all_vars_used, ret)
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

    def optimize(self, scale=1.0, debug=False):
        jakobi_list = []
        self.all_vars_used.clear()
        all_vars_opt = self.all_vars_used.keys()
        before = self.full_restrictions()
        #print('vv_',len(all_vars_opt), all_vars_opt)
        before = self.full_restrictions()  # must be called several times, as the reference chains may be created one by one
        #print('vv0',len(all_vars_opt), all_vars_opt)
        if debug:
            print('vars_to_opt', all_vars_opt)
            print('before', before)
        for (obj, vv) in all_vars_opt:
            #print('vv1',len(all_vars_opt))
            tmp = obj.getVar(vv)
            #print('vv2',len(all_vars_opt))
            obj.setVar(vv, tmp + 1)
            #print('vv3',len(all_vars_opt), all_vars_opt)
            after = self.full_restrictions()
            #print('vv4',len(all_vars_opt), all_vars_opt)
            #print('###',before,after)
            diff = list(map(operator.sub, after, before))
            #print('vv5',len(all_vars_opt))
            obj.setVar(vv, tmp)
            #print('vv6',len(all_vars_opt))
            jakobi_list.append(diff)
            #print('vv7',len(all_vars_opt))
        f_x = np.array(before)
        JK = np.array(jakobi_list, dtype=np.double)
        JK_t = JK.transpose()
        JK_t_i = np.linalg.pinv(JK_t, rcond=0.00001)
        delta = np.dot(JK_t_i, f_x)
        i = 0
        for (obj, vv) in all_vars_opt:
            obj.setVar(vv, obj.getVar(vv) - delta[i] * scale)
            i += 1
        check = self.full_restrictions()
        i = 0
        print('len of optimizing', len(check))
        isok = True
        for d in check:
            if abs(d) > 3:
                print('-opt-', i, d)
                isok = False
            i += 1
        return isok
        
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
                komma = False
                for i in range(len(eval(child_name))):
                    lleq = ll[1].split('=')
                    if len(lleq) == 1:
                        ret += replace_names(ll[1], child_name, i)
                    else:
                        right_side = replace_names(lleq[1], child_name, i)
                        left_side = replace_names(lleq[0], child_name, i)
                        rr = self.try_set_new(left_side, right_side)
                        if len(rr) > 0:
                            ret += rr + ','
                            komma = True
                if komma: ret = ret[:-1]
                ret += ']'
            elif ll[0] == 'between':
                ret = '['
                komma = False
                for i in range(1, len(eval(child_name))):
                    lleq = ll[1].split('=')
                    if len(lleq) == 1:
                        ret += replace_names(ll[1], child_name, i)
                    else:
                        right_side = replace_names(lleq[1], child_name, i)
                        left_side = replace_names(lleq[0], child_name, i)
                        #print('inbetw1',left_side,right_side,ret,'#')
                        rr = self.try_set_new(left_side, right_side)
                        if len(rr) > 0:
                            ret += rr + ','
                            komma = True
                        #print('inbetw2',left_side,right_side,ret,'#')
                if komma: ret = ret[:-1]
                ret += ']'
            elif ll[0] == 'min_all':
                # here references are not possible
                ret = 'min_all('+child_name +', lambda i: '
                ret += replace_names(ll[1], child_name)
                ret += ')'
                # for t in self.childs:
                #     print('l.top',t.top,self.top)
                # print('min_all', ret, eval(ret, dict(self=self, for_all=for_all, min_all=min_all, between=between)))
        # if len(ret)>2:
        #     print(self.__class__.__name__, ret)
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
        if self.TheCharacter is not None:
            w.create_text(self.getVar('left'), self.getVar('top'), anchor=NW, font=("Times New Roman", int(25), "bold"),
                      text=self.TheCharacter)

    def __init__(self, ch):
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved

        #fixed content not changed by other variables
        self.TheCharacter = ch
        BaseRules.__init__(self)

        #additional properties, not defined in priority
        self.add_property('height')
        self.add_property('width')

    def restrictions(self):
        return self.rule('top=bottom-height') + self.rule('right=left+width')

class Word(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved

        BaseRules.__init__(self)
        self.char_pos = -1

    def addCharacter(self, ch):
        l = Character(ch)
        if self.char_pos >= 0:
            self.childs.insert(self.char_pos, l)
            self.char_pos += 1
            self.clean_down()
            #for d in self.childs:
                 #print(d.TheCharacter, d.TheVars['left'], d.TheVars['right'])
        else:
            self.childs.append(l)
        return l

    def restrictions(self):
        ret = []
        if len(self.childs) > 0:
            ret += self.rule('firstchild.left=left')
            #print('vvv1',len(self.all_vars_used), self.all_vars_used, ret)
            ret += self.rule('min_all: child.top - top')
            #print('vvv2',len(self.all_vars_used), self.all_vars_used, ret)
            ret += self.rule('for_all: child.bottom=bottom')
            #print('vvv3',len(self.all_vars_used), self.all_vars_used, ret)
            ret += self.rule('between: rightchild.left=leftchild.right')
            #print('vvv4',len(self.all_vars_used), self.all_vars_used, ret)
            ret += self.rule('right=lastchild.right')
            #print('vvv5',len(self.all_vars_used), self.all_vars_used, ret)
        return ret

class Line(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved

        BaseRules.__init__(self)
        self.word_pos = -1

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
            if ll[-1].getVar('right') > self.getVar('right'):
                ToLong = ll.pop()
        return ToLong



class Page(BaseRules):
    def __init__(self):
        self.priority = [] #Later this should be syntactically improved
        BaseRules.__init__(self)

        #additional properties, not defined in priority
        self.add_property('top')
        self.add_property('bottom')
        self.add_property('left')
        self.add_property('right')
        self.line_pos = -1

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
            print('ctl')
            if add is not None:
                l.childs.insert(0, add)
                print('inserted')
            add = l.check_to_long()
        if add is not None:
            add.clean()
            actualLine = self.addLine()
            actualLine.childs.append(add)
            add = None
            print('nl')
        ToLong = None
        ll = self.childs
        if len(ll) > 0:
            if ll[-1].getVar('bottom') > self.getVar('bottom'):
                ToLong = ll.pop()
        print('ready',ToLong)
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
    global actualLine
    global actualWord
    print('button clicked', event)
    # mouse_coordinates= str(event.x) + ", " + str(event.y)
    # w.create_text(event.x, event.y, text = mouse_coordinates)
    x = event.x
    y = event.y
    for k, v in testpage.TheVars.items():
        print('  ', k, v, testpage.getVar(k))
    testpage.line_pos = -1
    l_count = 0
    for l in testpage.childs:
        print('Line')
        if l.top <= y < l.bottom:
            print('line clicked', l)
            testpage.line_pos = l_count
            l_count += 1
            actualLine = l
            w_count = 0
            l.word_pos = -1
            for ww in l.childs:
                print('Word')
                c_count = 0
                ww.char_pos = -1
                if ww.left <= x < ww.right:
                    print('word clicked', ww)
                    actualWord = ww
                    l.word_pos = w_count
                    for c in ww.childs:
                        print('Char')
                        if c.left <= x < c.right:
                            if x - c.left < c.right - x:
                                ww.char_pos = c_count
                            else:
                                ww.char_pos = c_count +1
                        c_count += 1

                w_count += 1
        else:
            l.word_pos = -1

    print('line', testpage.line_pos, 'word', actualLine.word_pos, 'char', actualWord.char_pos)
    printinfos()



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
    c = w.create_text(actualWord.top, actualWord.right, anchor=NW, font=("Times New Roman", int(25), "bold"),
                      text=event.char)
    print('key pressed', event, 'bounding', w.bbox(c))
    ch = event.char
    if ch == ' ':
        testpage.check_to_long()
        testpage.clean_down()
        actualWord = actualLine.addWord()
        l = actualWord.addCharacter(None)
        l.width = 0
        l.height = 0
        l.left = actualWord.right
        l.top = actualWord.top
        for pos in range(30):
            print('-----', pos, '---')
            if testpage.optimize(0.5):
                break
        w.delete("all")
        for d in testpage.get_all_self_and_childs():
            d.draw()
        full = testpage.full_restrictions(debug=0)
        #testpage.optimize()
        print('full', full)
        for l in testpage.childs:
            print(l,l.TheVars['top'],l.top,l.TheVars['bottom'],l.bottom)
        #printinfos()
    else:
        l = actualWord.addCharacter(ch)
        (left, top, right, bottom) = w.bbox(c)
        l.setVar('left', left)
        l.setVar('width', right-left)
        l.setVar('top', top)
        l.setVar('height', bottom-top)
        for _ in range(10):
            if testpage.optimize(0.9):
                break
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
