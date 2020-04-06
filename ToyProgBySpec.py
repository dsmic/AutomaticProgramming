#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:53:25 2020

@author: detlef
edr """

# pylint: disable=C0301, C0103, C0116, C0321, C0115, R0914, R0912, R0915, R1705, R1720, W0122, W0603, W0123, R1702

from tkinter import Tk, Canvas, mainloop, NW
from tokenize import tokenize
from io import BytesIO
import sympy as sym
# pylint: disable=W0611
from sympy import Min, Max, N # needed in sympy equations

def getVariable(name):
    return name.getVar

# takes a lambda function at the moment to compare two elements
    # e.g. between(ll,lambda a:ll[a].getVar('left')-ll[a-1].getVar('right'))
def min_all(ll, compare):
    #this is not really min, it must be possible to solve equation system with gradient
    #print('in min_all', opt)
    ret = None
    sum_neg = 0
    for i in range(0, len(ll)):
        t = compare(i)
        if ret is None or t < ret:
            ret = t
        #print('debugging',t)
        if t < 0:
            sum_neg += t
    if sum_neg < 0:
        return [sum_neg * 0.001] #this must be very small, as it should not break the rules for breaking lines
    else:
        return [ret]

def not_neg(ll, compare):
    #is more or less the same as min_all, but does not try to get close to zero if not neg
    ret = None
    sum_neg = 0
    for i in range(0, len(ll)):
        t = compare(i)
        if ret is None or t < ret:
            ret = t
        if t < 0:
            sum_neg += t
    if sum_neg < 0:
        return [0]
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
    priority = None # has to be overwritten by child instance variable
    all_vars_used = {}
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
        self.childs = []
        self.class_id = 'cid' + str(BaseRules.class_id_counter)
        self.solved_equations = None
        self.free_vars = None
        self.eqs_reduced = None
        BaseRules.classid_dict[self.class_id] = self
        BaseRules.class_id_counter += 1
        print(self.class_id, self.class_id_counter)
        #create the properties
        for l in self.priority:
            self.add_property(l)

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
        for c in self.childs:
            c.full_restrictions(debug=debug)
        #if debug and len(ret) > 0:
        #    print("full", type(self).__name__, len(ret), ret)
        #return ret

    def get_all_self_and_childs(self):
        ret = [self]
        for c in self.childs:
            ret += c.get_all_self_and_childs()
        return ret

    def solve_equations(self):
        self.clean_all_equations()
        self.full_restrictions()
        rrs = BaseRules.all_equations_rules+testpage.all_equations_min
        #print(rrs)
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
            #print('eq', eq)
            self.eqs_reduced.append(eq)
        #return (sr, free_vars, only_local_eq, self.eqs_reduced)

    def set_from_free_vars(self, free_vars_dict):
        if self.eqs_reduced is not None:
            for l, v in self.solved_equations.items():
                #print(l, v)
                v = eval(repr(v), free_vars_dict)
                #print(l, v)
                # they can be written into all parameters now
                free_vars_dict[repr(l)] = v
                vs = str(l).split('_')
                assert len(vs) == 2
                BaseRules.classid_dict[vs[0]].setVar(vs[1], N(v))
        #print('going into ', self.class_id)
        for wwww in self.childs:
            #print('wwww', wwww.class_id)
            wwww.set_from_free_vars(free_vars_dict)


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
                lll = ll[1].split('-') # must now be not a min_all: _____ - nochild_var
                symrule = 'Min('+replace_names_sympy(lll[0], child_name, 0)
                for i in range(1, len(eval(child_name))):
                    symrule += ',' + replace_names_sympy(lll[0], child_name, i)
                symrule += ')-'+replace_names_sympy(lll[1], child_name)
                all_min.append(symrule)
            if ll[0] == 'max_all':
                lll = ll[1].split('-') # must now be not a min_all: _____ - nochild_var
                symrule = 'Max('+replace_names_sympy(lll[0], child_name, 0)
                for i in range(1, len(eval(child_name))):
                    symrule += ',' + replace_names_sympy(lll[0], child_name, i)
                symrule += ')-'+replace_names_sympy(lll[1], child_name)
                all_min.append(symrule)
            if ll[0] == 'not_neg': # not sure if it should be exactly the same as min_all, we will see
                for i in range(len(eval(child_name))):
                    symrule = replace_names_sympy(ll[1], child_name, i)
                    all_checks.append(symrule)
        if BaseRules.all_equations_rules is not None:
            BaseRules.all_equations_rules += all_rules
            BaseRules.all_equations_checks += all_checks
            BaseRules.all_equations_min += all_min

    def restrictions(self):
        """
        are created from rules by adding the return value lists
        """
        # pylint: disable=R0201, W0101
        raise ValueError('This has to be overwritten by the child class')
        return []

    def add_child(self, a):
        self.childs.append(a)

    def child_type(self):
        # pylint: disable=R0201
        raise ValueError('has to be overwritten')


    def check_to_long3(self, solve_result):
        fv = {}
        for (l, v) in solve_result.items():
            #print(l, v)
            fv[repr(l)] = v
        for l in self.all_equations_checks:
            try:
                v = eval(l, fv)
                #print(l, v)
                if v > 0:
                    got = self.childs[-1].childs.pop()
                    self.childs[-1].eqs_reduced = None
                    # pylint: disable=E1111
                    l = self.child_type()
                    l.add_child(got)
                    self.add_child(l)
                    break
            except NameError:
                print('name not defined, but ok here')

class Character(BaseRules):
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

    def restrictions(self):
        self.rule('top=bottom-height')
        self.rule('right=left+width')

class Word(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'right', 'bottom', 'height'] #Later this should be syntactically improved

        BaseRules.__init__(self)
        self.char_pos = -1

    def addCharacter(self, ch):
        l = Character(ch)
        if self.char_pos >= 0:
            self.childs.insert(self.char_pos, l)
            self.char_pos += 1
            #self.clean_down()
            #for d in self.childs:
                 #print(d.TheCharacter, d.TheVars['left'], d.TheVars['right'])
        else:
            self.childs.append(l)
        return l

    def restrictions(self):
        #ret = []
        if len(self.childs) > 0:
            self.rule('firstchild.left=left')
            self.rule('max_all: child.height - height')
            self.rule('top + height - bottom')
            self.rule('for_all: child.bottom=bottom')
            self.rule('between: rightchild.left=leftchild.right')
            self.rule('right=lastchild.right')
        #return ret

class Line(BaseRules):
    def __init__(self):
        self.priority = ['top', 'left', 'right', 'bottom', 'freespace'] #Later this should be syntactically improved

        BaseRules.__init__(self)
        self.word_pos = -1


    def addWord(self):
        l = Word()
        self.add_child(l)
        return l

    def restrictions(self):
        #ret = []
        ll = self.childs
        if len(ll) > 0:
            self.rule('firstchild.left = left')
            if len(ll) > 1:
                self.rule('lastchild.right - right') # this is used to get 0 error if correct, but for optimizing we need direction if not correct
            self.rule('between: rightchild.left=leftchild.right+5 + freespace')
            self.rule('min_all: child.top - top')
            self.rule('for_all: child.bottom=bottom')
            self.rule('not_neg: -freespace')
        #return ret

class Page(BaseRules):
    def __init__(self):
        self.priority = [] #Later this should be syntactically improved
        BaseRules.__init__(self)

        #additional properties, not defined in priority
        self.add_property_setable('top')
        self.add_property_setable('bottom')
        self.add_property_setable('left')
        self.add_property_setable('right')
        self.line_pos = -1
        self.child_type = Line

    def addLine(self):
        l = Line()
        self.add_child(l)
        return l

    def restrictions(self):
        #ret = []
        # this must get good syntax later !!!!
        ll = self.childs
        if len(ll) > 0:
            self.rule('firstchild.top = top ')
            self.rule('for_all: child.left=left')
            self.rule('for_all: child.right=right')
            self.rule('between: rightchild.top = leftchild.bottom')

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



nextword = False
def key(event):
    global actualWord, nextword, actualLine
    c = w.create_text(actualWord.top, actualWord.right, anchor=NW, font=("Times New Roman", int(25), "bold"),
                      text=event.char)
    print('key pressed', event, 'bounding', w.bbox(c))
    ch = event.char
    if ch == ' ':
        testpage.clean_all_equations()
        testpage.full_restrictions(debug=0)
        solve_result = sym.solve(testpage.all_equations_rules + testpage.all_equations_min, dict=True)[0]
        for (vv, value) in solve_result.items():
            #print(vv, value)
            vs = str(vv).split('_')
            assert len(vs) == 2
            BaseRules.classid_dict[vs[0]].setVar(vs[1], value.evalf())

        lastLine = actualLine
        testpage.check_to_long3(solve_result)
        actualLine = testpage.childs[-1]

        actualWord = actualLine.childs[-1]
        nextword = True


        if lastLine != actualLine:
            print('new line now')
            lastLine.solve_equations()

            testpage.clean_all_equations()
            testpage.full_restrictions(debug=0)
            solve_result = sym.solve(testpage.all_equations_rules + testpage.all_equations_min, dict=True)[0]
            for (vv, value) in solve_result.items():
                #print(vv, value)
                vs = str(vv).split('_')
                assert len(vs) == 2
                BaseRules.classid_dict[vs[0]].setVar(vs[1], value.evalf())

        w.delete("all")
        for d in testpage.get_all_self_and_childs():
            d.draw()
    else:
        if nextword:
            actualWord = actualLine.addWord()
            nextword = False
        l = actualWord.addCharacter(ch)
        (left, top, right, bottom) = w.bbox(c)
        # pylint: disable=W0201
        l.left = left
        l.width = right-left
        l.top = top
        l.height = bottom-top

        actualWord.eqs_reduced = None # the Word was changed
        actualWord.solve_equations()
        #print(res_eq)

        testpage.clean_all_equations()
        testpage.full_restrictions(debug=0)

        solve_result = sym.solve(testpage.all_equations_rules + testpage.all_equations_min, dict=True)[0]


        fv = {}
        for (l, v) in solve_result.items():
            #print(l, v)
            fv[repr(l)] = v

        testpage.set_from_free_vars(fv)

        for new_string in BaseRules.all_equations_min:
            testtokens = tokenize(BytesIO(new_string.encode('utf-8')).readline)
            final_string = ''
            for tt in testtokens:
                ttr = tt.string
                if tt.type == 1:
                    if sym.sympify(ttr) in solve_result:
                        ttr = solve_result[sym.sympify(ttr)].evalf()

                    final_string += str(ttr)
                elif tt.type not in [59, 57]:
                    final_string += ttr

        for (vv, value) in solve_result.items():
            vs = str(vv).split('_')
            assert len(vs) == 2
            BaseRules.classid_dict[vs[0]].TheVars[vs[1]] = value.evalf()

        w.delete("all")
        for d in testpage.get_all_self_and_childs():
            d.draw()

w.bind('<Button-1>', click)
master.bind('<Key>', key)
mainloop()
