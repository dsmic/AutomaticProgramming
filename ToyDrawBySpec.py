#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:53:25 2020

@author: detlef
edr """

# pylint: disable=C0301, C0103, C0116, C0321, C0115, R0914, R0912, R0915, R1705, R1720, W0122, W0603, W0123, R1702

import tkinter as tk
from tkinter import Tk, Canvas, mainloop, NW
from tokenize import tokenize
from io import BytesIO

import os
import subprocess
import shutil
import importlib
import math

import sympy as sym
# pylint: disable=W0611
from sympy import Min, Max, N # needed in sympy equations

import drawmodules


show_called_file = True

def getVariable(name):
    return name.getVar

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
    # pylint: disable=R0902
    priority = None # has to be overwritten by child instance variable

    class_id_counter = 0
    classid_dict = {}

    all_equations_rules = []
    all_equations_checks = []
    all_equations_min = []

    @classmethod
    def clean_all_equations(cls):
        BaseRules.all_equations_rules = []
        BaseRules.all_equations_checks = []
        BaseRules.all_equations_min = []
    @classmethod
    def stop_all_equations(cls):
        BaseRules.all_equations_rules = None
        BaseRules.all_equations_checks = None
        BaseRules.all_equations_min = None

    properties_setable = [] # all this have a value set by hand

    def draw(self):
        pass

    def add_property(self, l):
        classname = self.__class__.__name__
        s1 = 'def gvar_'+l+'(self): return self.getVar("'+l+'")'
        s2 = 'def svar_'+l+'(self, x): return self.setVar("'+l+'",x)' # not sure if this can be used later?!
        s3 = classname+'.' + l +' = property(gvar_'+l+',svar_'+l+')'

        exec(s1)
        exec(s2)
        exec(s3)

    def add_property_setable(self, l):
        self.add_property(l)
        BaseRules.properties_setable.append(self.class_id + '_' + l)

    def __init__(self):
        #manageing variables (not used in later syntax)
        self.TheVars = {} #contains the variables from priority
        self.clean()
        self.childs = [[]]
        self.class_id = 'cid' + str(BaseRules.class_id_counter)
        self.solved_equations = None
        self.free_vars = None
        self.eqs_reduced = None
        BaseRules.classid_dict[self.class_id] = self
        BaseRules.class_id_counter += 1
        print(self.class_id, self.class_id_counter, self.__class__.__name__)
        #create the properties
        for l in self.priority:
            self.add_property(l)
        self.RestrictionsList = [] #executed by BaseRules class
        self.ParentRestrictionsList = []

    def clean(self):
        for l in self.priority:
            self.TheVars[l] = 0

    def getVar(self, name):
        return self.TheVars[name]

    def setVar(self, name, value):
        self.TheVars[name] = value

    def full_restrictions(self, debug=0):
        if self.eqs_reduced is not None:
            BaseRules.all_equations_rules += self.eqs_reduced
            return
        self.restrictions()
        for cc in self.childs:
            for c in cc:
                c.full_restrictions(debug=debug)

    def get_all_self_and_childs(self):
        ret = [self]
        for cc in self.childs:
            for c in cc:
                ret += c.get_all_self_and_childs()
        return ret

    def solve_equations(self):
        self.clean_all_equations()
        self.full_restrictions()
        rrs = BaseRules.all_equations_rules+BaseRules.all_equations_min
        # this are the vars of childs
        list_vars = []
        # this are vars of self, they could be keep unsolved, as they are the parameters
        list_vars_self = []
        for rrr in rrs:
            testtokens = tokenize(BytesIO(rrr.encode('utf-8')).readline)
            for tt in testtokens:
                #print(tt)
                if tt.type == 1:
                    if tt.string.split('_')[0] == self.class_id and tt.string not in list_vars+list_vars_self:
                        list_vars_self.append(tt.string)
                    elif tt.string[:3] == 'cid' and tt.string not in list_vars:
                        list_vars.append(tt.string)

        sr = sym.solve(rrs, list_vars+list_vars_self, dict=True)[0]
        free_vars = [l for l in list_vars+list_vars_self if sym.sympify(l) not in sr]

        # all parameters below can be set with solved_equations from free_vars now
        self.solved_equations = sr
        self.free_vars = free_vars

        only_local_eq = {}
        for (s, v) in self.solved_equations.items():
            if repr(s) in list_vars_self: #[sym.sympify(l) for l in list_vars_self]:
                only_local_eq[s] = v
        self.eqs_reduced = []
        for (s, v) in only_local_eq.items():
            eq = repr(s)+'-('+repr(v)+')'
            self.eqs_reduced.append(eq)

    def set_from_free_vars(self, free_vars_dict):
        if self.eqs_reduced is not None:
            for l, v in self.solved_equations.items():
                v = eval(repr(v), free_vars_dict)
                # they can be written into all parameters now
                free_vars_dict[repr(l)] = v
                vs = str(l).split('_')
                assert len(vs) == 2
                #print('setting',vs[0], vs[1], N(v))
                BaseRules.classid_dict[vs[0]].setVar(vs[1], N(v))
        for wwwww in self.childs:
            for wwww in wwwww:
                wwww.set_from_free_vars(free_vars_dict)


    def rule(self, rulestring, child_name='self.childs[0]'):
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
        def replace_names_sympy(string, child_name, i=None):
            transform_to_null = string.split('=')
            if len(transform_to_null) == 2: #sympy must have equation without written =0
                string = transform_to_null[0] + '-(' + transform_to_null[1] +')'
            string = string.strip()
            testtokens = tokenize(BytesIO(string.encode('utf-8')).readline)
            new_string = ''
            afterdot = False
            for tt in testtokens:
                if tt.type == 1:
                    # here string replacement will be possible
                    ttt = tt.string
                    if afterdot:
                        new_string += ttt
                        afterdot = False
                    elif ttt == 'firstchild':
                        new_string += eval(child_name+'[0].class_id', dict(self=self))
                    elif ttt == 'lastchild':
                        new_string += eval(child_name+'[-1].class_id', dict(self=self))
                    elif ttt in ('child', 'rightchild'):
                        if i is None:
                            new_string += child_name +"[i]"
                        else:
                            new_string += eval(child_name+'['+str(i)+'].class_id', dict(self=self))
                    elif ttt == 'leftchild':
                        new_string += eval(child_name+'['+str(i-1)+'].class_id', dict(self=self))
                    else:
                        new_string += eval('self.class_id', dict(self=self)) + '_' + ttt
                    afterdot = False
                elif tt.type not in [59, 57]:
                    tt_r = tt.string
                    if tt.string == '.': afterdot = True
                    if tt_r == '.':
                        tt_r = '_'
                    new_string += tt_r

            testtokens = tokenize(BytesIO(new_string.encode('utf-8')).readline)
            final_string = ''
            for tt in testtokens:
                ttr = tt.string
                if tt.type == 1:
                    if ttr in BaseRules.properties_setable:
                        tts = ttr.split('_')
                        use = BaseRules.classid_dict[tts[0]]
                        ttr = eval('use.'+tts[1], dict(use=use))

                    final_string += str(ttr)
                elif tt.type not in [59, 57]:
                    final_string += ttr

            return final_string

        ll = rulestring.split(':')
        all_rules = []
        all_checks = []
        all_min = []
        if len(ll) == 1:
            symrule = replace_names_sympy(rulestring, child_name)
            if '>' in ll[0] or '<' in ll[0]:
                all_checks.append(symrule)
            else:
                all_rules.append(symrule)
        else:
            if ll[0] == 'for_all':
                for i in range(len(eval(child_name))):
                    symrule = replace_names_sympy(ll[1], child_name, i)
                    all_rules.append(symrule)
            if ll[0] == 'between':
                for i in range(1, len(eval(child_name))):
                    symrule = replace_names_sympy(ll[1], child_name, i)
                    all_rules.append(symrule)
            if ll[0] == 'min_all':
                lll = ll[1].split('=') # must now be not a min_all: _____ - nochild_var
                symrule = 'Min('+replace_names_sympy(lll[0], child_name, 0)
                for i in range(1, len(eval(child_name))):
                    symrule += ',' + replace_names_sympy(lll[0], child_name, i)
                symrule += ')-'+replace_names_sympy(lll[1], child_name)
                all_min.append(symrule)
            if ll[0] == 'max_all':
                lll = ll[1].split('=') # must now be not a min_all: _____ - nochild_var
                symrule = 'Max('+replace_names_sympy(lll[0], child_name, 0)
                for i in range(1, len(eval(child_name))):
                    symrule += ',' + replace_names_sympy(lll[0], child_name, i)
                symrule += ')-'+replace_names_sympy(lll[1], child_name)
                all_min.append(symrule)

        if BaseRules.all_equations_rules is not None:
            BaseRules.all_equations_rules += all_rules
            BaseRules.all_equations_checks += all_checks
            BaseRules.all_equations_min += all_min

    def restrictions(self):
        """
        are created from rules by adding the return value lists
        """
        for l in self.RestrictionsList:
            self.rule(l)
        for l in self.childs:
            for ll in l:
                for lll in ll.ParentRestrictionsList:
                    self.rule(lll)

    def add_child(self, a):
        self.childs[0].append(a)

    def child_type(self):
        # pylint: disable=R0201
        raise ValueError('has to be overwritten')


    def check_to_long3(self):
        ret = False
        self.clean_all_equations()
        self.full_restrictions()
        solve_result = sym.solve(self.all_equations_rules + self.all_equations_min, dict=True)[0]
        fv = {}
        for (l, v) in solve_result.items():
            #print(l, v)
            fv[repr(l)] = v
        for l in self.all_equations_checks:
            try:
                v = eval(l, fv)
                #print(l, v)
                if not v:
                    got = self.childs[0][-1].childs[0].pop()
                    self.childs[0][-1].eqs_reduced = None
                    # pylint: disable=E1111
                    l = self.child_type()
                    l.add_child(got)
                    self.add_child(l)
                    ret = True
                    break
            except NameError:
                print('name not defined, but ok here')
        return ret

    def full_set(self):
        self.clean_all_equations()
        self.full_restrictions()
        solve_result = sym.solve(self.all_equations_rules + self.all_equations_min, dict=True)[0]

        fv = {}
        for (l, v) in solve_result.items():
            print(l, v)
            fv[repr(l)] = v
            vs = str(l).split('_')
            assert len(vs) == 2
            self.classid_dict[vs[0]].TheVars[vs[1]] = v.evalf()

        self.set_from_free_vars(fv)

    def clickCheck(self, pos):
        # pylint: disable=E1101
        x = pos.x
        y = pos.y
        print(x, y, self.left, self.right, self.top, self.bottom)
        ret = []
        for l in self.childs[0]:
            ret += l.clickCheck(pos)
        if self.left <= x < self.right and self.top <= y < self.bottom:
            ret += [self]
        return ret

    def clicktree(self, item):
        if item == self:
            return [self]
        for l in self.childs[0]:
            r = l.clicktree(item)
            if r is not None:
                return r + [self]
        return None

# class Character(BaseRules):
#     def draw(self):
#         #print(self, 'char draw', self.TheCharacter, round(self.getVar('left')), round(self.getVar('top')), round(self.getVar('right')), round(self.getVar('bottom')))
#         if self.TheCharacter is not None:
#             w.create_text(self.getVar('left'), self.getVar('top'), anchor=NW, font=("Times New Roman", int(25), "bold"),
#                           text=self.TheCharacter)

#     def __init__(self, ch):
#         # this my be created automatically from rules, buth than fixed ones from add_property must stay by hand I think
#         # Maybe should be kept like this, as this helps organizing what to use (as defining vars in other programming languages)
#         self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved

#         #fixed content not changed by other variables
#         self.TheCharacter = ch
#         BaseRules.__init__(self)

#         #additional properties, not defined in priority
#         self.add_property_setable('height')
#         self.add_property_setable('width')
#         self.RestrictionsList = ['top=bottom-height',
#                                  'right=left+width']

# class Word(BaseRules):
#     def __init__(self):
#         self.priority = ['top', 'left', 'right', 'bottom', 'height'] #Later this should be syntactically improved

#         BaseRules.__init__(self)
#         self.char_pos = -1
#         self.RestrictionsList = ['firstchild.left=left',
#                                  'max_all: child.height = height',
#                                  'top + height - bottom',
#                                  'for_all: child.bottom=bottom',
#                                  'between: rightchild.left=leftchild.right',
#                                  'right=lastchild.right']
#     def addCharacter(self, ch):
#         l = Character(ch)
#         if self.char_pos >= 0:
#             self.childs[0].insert(self.char_pos, l)
#             self.char_pos += 1
#         else:
#             self.childs[0].append(l)
#         return l

# class Line(BaseRules):
#     def __init__(self):
#         self.priority = ['top', 'left', 'right', 'bottom', 'freespace'] #Later this should be syntactically improved

#         BaseRules.__init__(self)
#         self.word_pos = -1
#         self.RestrictionsList = ['firstchild.left = left',
#                                  'between: rightchild.left=leftchild.right+5 + freespace',
#                                  'min_all: child.top = top',
#                                  'for_all: child.bottom=bottom',
#                                  '0<freespace']

#     def addWord(self):
#         l = Word()
#         self.add_child(l)
#         return l

#     def restrictions(self):
#         if len(self.childs[0]) > 1: #this rule only applies, if there are two or more words in a line, otherwize it is not possible to match left and right!
#             self.rule('lastchild.right - right') # this is used to get 0 error if correct, but for optimizing we need direction if not correct
#         BaseRules.restrictions(self) # call RestrictionsList

# class Page(BaseRules):
#     def __init__(self):
#         self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved
#         BaseRules.__init__(self)

#         self.line_pos = -1
#         self.child_type = Line
#         self.RestrictionsList = ['firstchild.top = top ',
#                                  'for_all: child.left=left',
#                                  'for_all: child.right=right',
#                                  'between: rightchild.top = leftchild.bottom']

#     def addLine(self):
#         l = Line()
#         self.add_child(l)
#         return l

class MenuItem(BaseRules):
    def draw(self):
        #print(self, 'char draw', self.TheCharacter, round(self.getVar('left')), round(self.getVar('top')), round(self.getVar('right')), round(self.getVar('bottom')))
        if self.TheCharacter is not None:
            w.create_text(self.getVar('left'), self.getVar('top'), anchor=NW, font=("Times New Roman", int(25), "bold"),
                          text=self.TheCharacter)

    def __init__(self, ch):
        # this my be created automatically from rules, buth than fixed ones from add_property must stay by hand I think
        # Maybe should be kept like this, as this helps organizing what to use (as defining vars in other programming languages)
        self.priority = ['top', 'left', 'right', 'bottom'] #Later this should be syntactically improved

        #fixed content not changed by other variables
        self.TheCharacter = ch
        BaseRules.__init__(self)

        #additional properties, not defined in priority
        self.add_property_setable('height')
        self.add_property_setable('width')
        c = w.create_text(600, 600, anchor=NW, font=("Times New Roman", int(25), "bold"),
                          text=ch)
        (x1, y1, x2, y2) = w.bbox(c)
        self.height = y2-y1
        self.width = x2-x1
        print(w.bbox(c), self.height, self.width)
        self.name = None
        self.RestrictionsList = ['top=bottom-height', 'right=left+width']


    def addMenu(self, name, ItemList, ParentRestrictionsList=None, horizontal=False):
        l = Menu(self.name + '_' + name, ItemList, horizontal, ParentRestrictionsList)
        self.add_child(l)
        return l


class Menu(BaseRules):
    def __init__(self, name, ItemList, horizontal=True, ParentRestrictionsList=None):
        self.priority = ['top', 'right', 'left', 'bottom'] #Later this should be syntactically improved

        BaseRules.__init__(self)
        # self.add_property_setable('top')
        # self.add_property_setable('right')
        self.menuname = name
        if horizontal:
            self.RestrictionsList = ['lastchild.right = right',
                                     'between: rightchild.left=leftchild.right+5',
                                     'min_all: child.top = top',
                                     'for_all: child.bottom=bottom']
        else:
            self.RestrictionsList = ['firstchild.top = top',
                                     'between: rightchild.top=leftchild.bottom+5',
                                     'for_all: child.right = right',
                                     'lastchild.bottom = bottom'
                                     ]
        for l in ItemList:
            self.addMenuItem(l)
        self.ParentRestrictionsList = ParentRestrictionsList

    def addMenuItem(self, name):
        l = MenuItem(name)
        l.name = self.menuname + '_' + name
        self.add_child(l)
        return l

# testpage = Page()

# testpage.RestrictionsList.append('left')
# testpage.RestrictionsList.append('top-200')
# testpage.RestrictionsList.append('right-400')
# testpage.RestrictionsList.append('bottom-400')

# actualLine = testpage.addLine()
# actualWord = actualLine.addWord()

# Toy Manager
master = Tk()
canvas_width = 2000
canvas_height = 1000
w = Canvas(master, width=canvas_width, height=canvas_height)
w.pack()

def click(event):
    importlib.reload(drawmodules)
    global actualLine
    global actualWord
    print('button clicked', event)
    # mouse_coordinates= str(event.x) + ", " + str(event.y)
    # w.create_text(event.x, event.y, text = mouse_coordinates)
    # x = event.x
    # y = event.y
    # for k, v in testpage.TheVars.items():
    #     print('  ', k, v, testpage.getVar(k))
    # testpage.line_pos = -1
    # l_count = 0
    # for l in testpage.childs[0]:
    #     print('Line')
    #     if l.top <= y < l.bottom:
    #         print('line clicked', l)
    #         testpage.line_pos = l_count
    #         l_count += 1
    #         actualLine = l
    #         w_count = 0
    #         l.word_pos = -1
    #         for ww in l.childs[0]:
    #             print('Word')
    #             c_count = 0
    #             ww.char_pos = -1
    #             if ww.left <= x < ww.right:
    #                 print('word clicked', ww)
    #                 actualWord = ww
    #                 l.word_pos = w_count
    #                 for c in ww.childs[0]:
    #                     print('Char')
    #                     if c.left <= x < c.right:
    #                         if x - c.left < c.right - x:
    #                             ww.char_pos = c_count
    #                         else:
    #                             ww.char_pos = c_count +1
    #                     c_count += 1

    #             w_count += 1
    #     else:
    #         l.word_pos = -1
    # print('line', testpage.line_pos, 'word', actualLine.word_pos, 'char', actualWord.char_pos)


    # printinfos()
    r = menu.clickCheck(event)
    if len(r) > 0:
        clicktree = menu.clicktree(r[0])
        print('clicktree', clicktree)
        print('clicked', r[0].name)
        try:
            if show_called_file:
                nf = 'drawmodules/'+r[0].name+'.py'
                subprocess.call(["spyder", nf])
            exec('drawmodules.'+r[0].name+".call(clicktree)")
        except AttributeError:
            nf = 'drawmodules/'+r[0].name+'.py'
            if not os.path.exists(nf):
                shutil.copyfile('drawmodules/template', nf)
                subprocess.call(["spyder", nf])
    else:
        print('handling usual click')
    menu.full_set()
    w.delete("all")
    for dd in menu.get_all_self_and_childs():
        dd.draw()

# def printinfos():
#     print('Page')
#     for k, v in testpage.TheVars.items():
#         print('  ', k, v, testpage.getVar(k))
#     for l in testpage.childs[0]:
#         print('Line')
#         for k, v in l.TheVars.items():
#             print('     ', k, v, l.getVar(k))
#         for ww in l.childs[0]:
#             print('Word')
#             for k, v in ww.TheVars.items():
#                 print('        ', k, v, ww.getVar(k))
#             for c in ww.childs[0]:
#                 print('Char')
#                 for k, v in c.TheVars.items():
#                     print('            ', k, v, c.getVar(k))

# nextword = False
# def key(event):
#     global actualWord, nextword, actualLine
#     c = w.create_text(actualWord.top, actualWord.right, anchor=NW, font=("Times New Roman", int(25), "bold"),
#                       text=event.char)
#     print('key pressed', event, 'bounding', w.bbox(c))
#     ch = event.char
#     if ch == ' ':
#         lastLine = actualLine
#         nextword = True
#         if testpage.check_to_long3():
#             print('new line now')
#             actualLine = testpage.childs[0][-1]
#             actualWord = actualLine.childs[0][-1]
#             lastLine.solve_equations()
#             testpage.full_set()

#         w.delete("all")
#         for dd in testpage.get_all_self_and_childs() + menu.get_all_self_and_childs():
#             dd.draw()
#     else:
#         if nextword:
#             actualWord = actualLine.addWord()
#             nextword = False
#         l = actualWord.addCharacter(ch)
#         (left, top, right, bottom) = w.bbox(c)
#         # pylint: disable=W0201
#         l.left = left
#         l.width = right-left
#         l.top = top
#         l.height = bottom-top

#         actualWord.eqs_reduced = None # the Word was changed
#         actualWord.solve_equations()
#         testpage.full_set()

#         w.delete("all")
#         for dd in testpage.get_all_self_and_childs() + menu.get_all_self_and_childs():
#             dd.draw()

lastpress = None
lastpoints = None
last_line_properties = None
markedpoint = None

lastdirect = 0

def abst(a,b):
    print(a.x,a.y,b.x,b.y)
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2)

def kreuz(a,b,c):
    # sign of kreuzprodukt
    return math.copysign(1,(c.x-a.x)*(b.y-a.y)-(c.y-a.y)*(b.x-a.x))
def direct(a,b):
    return math.atan2(a.y-b.y, a.x-b.x)

def mouserelease(event):
    global lastdirect, last_line_properties, lastpoints
    print('release',event)
    #******************************
    # this is working for draw the full line
    # pointflat = [a for l in lastpoints for a in (l.x,l.y)]
    # w.create_line(*pointflat, smooth=True, splinesteps=3)
    # print(pointflat)
    #******************************
    if abs(event.x - lastpress.x)< 5 and abs(event.y-lastpress.y)<5:
        click(event)
    # calculate line properties
    start = lastpoints[0]
    end = lastpoints[-1]
    lp = lastpoints[0]
    ssum = 0
    for l in lastpoints[1:]:
        ssum += abst(l,lp)
        lp = l
    krum = 0
    for i in range(2,len(lastpoints)):
        c = abst(lastpoints[i-2],lastpoints[i])
        if c > 0:
            a = abst(lastpoints[i-1],lastpoints[i])
            b = abst(lastpoints[i-2],lastpoints[i-1])
            s = (a+b+c)/2
            try:
                print(a,b,c,s,  2/c*math.sqrt(s*(s-a)*(s-b)*(s-c)))
                krum += 2/c*math.sqrt(s*(s-a)*(s-b)*(s-c)) * kreuz(lastpoints[i-2],lastpoints[i-1],lastpoints[i])
                print(kreuz(lastpoints[i-2],lastpoints[i-1],lastpoints[i]), krum)
            except ValueError:
                print('ValueError')
    thisdirect = direct(start,end)
    # print(abs(math.modf((thisdirect-lastdirect)/math.pi)[0]))
    # print(abs(abs(math.modf((thisdirect-lastdirect)/math.pi)[0])-0.5)*2)
    ct = point((start.x+end.x)/2, (start.y+end.y)/2)
    if ssum > 0:
        kr = krum/ssum
    else:
        kr = 0
    line_properties = {'start': start, 'end': end, 'center': ct, 'length': ssum, 'curvature': kr, 'direction': thisdirect, 
                       'parallel_to_last': abs(abs(math.modf((thisdirect-lastdirect)/math.pi)[0])-0.5)*2,
                       'pointlist': lastpoints}
    done = drawmodules.draw_line_ready.call(line_properties, last_line_properties)
    if done:
        last_line_properties = None
        lastdirection = None
        w.delete("all")
        lastpoints = None
    else:
        last_line_properties = line_properties.copy()
        lastdirect = thisdirect
    for d in draw_objects:
        print('draw', d)
        d.draw()
        
def mousepress(event):
    global lastpress, lastpoints
    lastpress = event
    lastpoints = [event]
    print('press',event)

def mousemove(event):
    global lastpoints
    # if abst(event,lastpoints[-1]) < 100:
    #     return
    w.create_line(lastpoints[-1].x, lastpoints[-1].y, event.x, event.y)
    lastpoints += [event]
    print('move',event)
class point():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        
class draw_point():
    def __init__(self, cx, cy):
        self.x = cx
        self.y = cy
    
    def draw(self):
        cx = self.x
        cy = self.y
        w.create_line(cx-5, cy-5, cx+5, cy+5, fill="red")
        w.create_line(cx-5, cy+5, cx+5, cy-5, fill="red")

class draw_line():
    def __init__(self, sp, ep):
        self.sp = sp
        self.ep = ep
    def draw(self):
        w.create_line(self.sp.x, self.sp.y, self.ep.x, self.ep.y, fill="red")
        
class draw_polygon():
    def __init__(self, pg):
        self.pg = pg
    
    def draw(self):
        pg = self.pg
        if len(pg) >= 4:
            w.create_line(*pg, fill="red")
        
class draw_circle():
    def __init__(self, cp, radius):
        self.cp = cp
        self.radius = radius
    def draw(self):
        cp=self.cp
        radius=self.radius
        w.create_oval(cp.x - radius, cp.y - radius, cp.x +  radius, cp.y + radius, outline='red')

def find_point_near(point, dist=None):
    mindist = None
    minpoint = None
    for p in draw_objects:
        if isinstance(p, draw_point):
            if mindist is None:
                mindist = abst(point, p)
                minpoint = p
            else:
                ab = abst(point, p)
                if ab < mindist:
                    mindist = ab
                    minpoint = p
    if mindist is None:
        return None
    if dist is None or mindist < dist:
        return minpoint
    return None

def intersect_line_line(l1, l2):
    l1sx = l1.sp.x
    l1sy = l1.sp.y
    l1ex = l1.ep.x
    l1ey = l1.ep.y
    l2sx = l2.sp.x
    l2sy = l2.sp.y
    l2ex = l2.ep.x
    l2ey = l2.ep.y
    g1 = str(l1sx) + ' + x1 * '+ str(l1ex-l1sx) + '-('+   str(l2sx) + ' + x2 * '+ str(l2ex-l2sx) +')'
    g2 = str(l1sy) + ' + x1 * '+ str(l1ey-l1sy) + '-('+   str(l2sy) + ' + x2 * '+ str(l2ey-l2sy) +')'
    print(g1)
    print(g2)
    try:
        r = sym.solve([g1,g2])
        print(r)
        x1 = N(r[sym.sympify('x1')])
        x2 = N(r[sym.sympify('x2')])
        print(l1sx+ x1 * (l1ex-l1sx))
        print(l1sy+ x1 * (l1ey-l1sy))
        if 0<=x1<=1 and 0<=x2<=1:
            return [point(l1sx+ x1 * (l1ex-l1sx), l1sy+ x1 * (l1ey-l1sy))]
    except TypeError:
        print('not solveable')
    except KeyError:
        print('not solveable')
    return []

def intersect_line_circle(l1, c2):
    ret = []
    l1sx = l1.sp.x
    l1sy = l1.sp.y
    l1ex = l1.ep.x
    l1ey = l1.ep.y
    c2x = c2.cp.x
    c2y = c2.cp.y
    radius = c2.radius
    g1 = '(' + str(l1sx) + ' + x1 * '+ str(l1ex-l1sx) + '-' + str(c2x) +')**2 + ' + '(' + str(l1sy) + ' + x1 * '+ str(l1ey-l1sy) + '-' + str(c2y) +')**2 -' + str(radius) + '**2'
    print(g1)
    try:
        r = sym.solve(g1)
    except TypeError:
        return ret
    print(r)
    for r1 in r:
        try:
            x1 = N(r1)
            print('lösungen',x1)
            if 0<=r1<=1:
                ret.append(point(l1sx+ x1 * (l1ex-l1sx), l1sy+ x1 * (l1ey-l1sy)))
        except TypeError:
            pass
    return ret
            
def intersect_circle_circle(c1, c2):
    ret = []
    c1x = c1.cp.x
    c1y = c1.cp.y
    radius1 = c2.radius
    c2x = c2.cp.x
    c2y = c2.cp.y
    radius2 = c2.radius
    g1 = '(  x  - ' + str(c1x) +')**2 + ( y - '  + str(c1y) + ')**2 - ' + str(radius1)
    g2 = '(  x  - ' + str(c2x) +')**2 + ( y - '  + str(c2y) + ')**2 - ' + str(radius2)
    print(g1)
    print(g2)
    try:
        r = sym.solve([g1,g2])
    except TypeError:
        return ret
    print(r)
    for r1 in r:
        try:
            x = N(r1[sym.sympify('x')])
            print('lösungen x',x)
            y = N(r1[sym.sympify('y')])
            print('lösungen y',y)
            ret.append(point(x,y))
        except TypeError:
            pass
    return ret
            
   
    

def find_intersections():
    point_list = []
    tmp_draw = draw_objects[:]
    print('draw', draw_objects)
    while len(tmp_draw) > 1:
        o1 = tmp_draw.pop()
        if isinstance(o1, draw_line):
            for o2 in tmp_draw:
                if isinstance(o2, draw_line):
                    point_list += intersect_line_line(o1,o2)
                if isinstance(o2, draw_circle):
                    point_list += intersect_line_circle(o1,o2)
        if isinstance(o1, draw_circle):
            for o2 in tmp_draw:
                if isinstance(o2, draw_line):
                    point_list += intersect_line_circle(o2,o1)
                if isinstance(o2, draw_circle):
                    point_list += intersect_circle_circle(o1,o2)
    return point_list

draw_objects = []
    
w.bind('<ButtonRelease-1>', mouserelease)
w.bind('<ButtonPress-1>', mousepress)
w.bind('<B1-Motion>', mousemove)

#master.bind('<Key>', key)

menu = Menu("main", ['Datei', 'Edit'], horizontal=True)
menu.RestrictionsList.append('right-1990')
menu.RestrictionsList.append('top-10')

menu.full_set()
w.delete("all")
for d in menu.get_all_self_and_childs():
    d.draw()

w.create_rectangle(10,10,100,100)
w.create_arc(-100,-100,100,100, extent=359.99, style=tk.ARC)
mainloop()

sp = point(2,4)
ep = point(60,70)
sp2 = point(20,5)
ep2 = point(6,7.5)
l1 = draw_line(sp,ep)
l2 = draw_line(sp2,ep2)
c1 = draw_circle(sp,5)
c2 = draw_circle(sp2,5)
o=intersect_line_line(l1,l2)
print(o)
o=intersect_line_circle(l1,c2)
print(o)
o=intersect_circle_circle(c1,c2)
print(o)