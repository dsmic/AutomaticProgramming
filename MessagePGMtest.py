#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:38:16 2020

@author: detlef
"""

class MyList(list):
    def remove_set(self, rm_set):
        for el in rm_set:
            if el in self:
                self.remove(el)

class Meta(type):
     def __new__(cls, name, bases, dct):
         x = super().__new__(cls, name, bases, dct)
         print('META',name, bases, dct, x.msg_string())
         if x.msg_string() is not None:
             x.create_msgs[x.msg_string()] = x
         return x        

class BaseClassMSG(metaclass=Meta):
    create_msgs = {}
    def __init__(self):
        self.childs = [] # this are the childs, which get the messages
        self.subobjects = [] # objects, for with the childs are handled
        pass
    
    def process_msgs(self, msgs):
        print(type(self), 'process_msgs', msgs)
        goon = True
        while True:
            (msgs_all_handled, msgs_all_added, goon) = self.msgs_receive(msgs, goon)
            if len(self.childs) == 0:
                return  (msgs_all_handled, msgs_all_added) # no childs, no further handling
    
            msgs_to_childs = MyList(msgs[:])
            msgs_to_childs.remove_set(msgs_all_handled)
            
            if len(self.subobjects) > 0:
                subobjects = self.subobjects
                for sub in subobjects:
                    sub.childs = self.childs # prepare the subobjects with the childs
            else:
                subobjects = self.childs
                
            
            for child_object in subobjects:
                (msgs_handled, msgs_added) = child_object.process_msgs(msgs_to_childs + msgs_all_added) #it might be a good idea, that msgs_all_added is passed, than the subchilds can proces output from the subchilds before ?!
                msgs_all_added.remove_set(msgs_handled)
                msgs_all_added += msgs_added
                msgs_all_handled.update(msgs_handled)
            msgs_to_childs.remove_set(msgs_all_handled)
            msgs = msgs_to_childs + msgs_added
            if not goon:
                break
            
        # call self.msgs_receive a second time? with msgs_all_added and leave the rest????
        (msgs_second_handled, msgs_second_added, goon) = self.msgs_receive(msgs_to_childs + msgs_all_added, goon)
        
        rest_added_msgs = MyList(msgs_all_added + msgs_second_added)
        msgs_all_handled.update(msgs_second_handled)
        rest_added_msgs.remove_set(msgs_second_handled)
              
        return (msgs_all_handled, rest_added_msgs)
        
    
    # overwrite
    # remove the messages from msg, which are handled (maybe also by the childs??), the rest is kept
    def msgs_receive(self, msgs, goon):
        print(type(self), 'msgs_received', msgs, goon)
        return (set(), MyList([]), False) #return msgs set that where handled and new Messages
    
    def msg_string():
        return None
    

class TestMessage(BaseClassMSG):
    def msgs_receive(self, msgs, goon):
        print(type(self), 'msgs_received', msgs, goon)
        return (set(), MyList([]), False) #return msgs set that where handled and new Messages

    def msg_string():
        return 'create_TestMessage'
        
class TestAdd(BaseClassMSG):
    def __init__(self):
        super().__init__()
        self.summe = 0
        
    def msgs_receive(self, msgs, goon):
        print(type(self), 'msgs_received', msgs, goon)
        #summe = 0
        handled = set()
        for msg in msgs:
            (msg_type, value) = msg
            if msg_type == 'init':
                self.summe = 0
            if msg_type == 'float':
                handled.add(msg)
                self.summe += value
        if len(handled) == 0:
            return (handled, [], False)
        return (handled, [('result', self.summe)], False)

    def msg_string():
        return 'create_TestAdd'

class testList(BaseClassMSG):
    def __init__(self):
        super().__init__()
        self.list_objects = []
        self.position = 0
    
    def msgs_receive(self, msgs, goon):
        print(type(self), 'msgs_received', msgs, goon)
        handled = set()
        msgs_to_childs = []
        if self.position == 0:
            msgs_to_childs.append(('init', 0))
        for msg in msgs:
            (msg_type, value) = msg
            if msg_type == 'init':
                self.list_objects = value
                self.position = 0
            if goon and msg_type == 'result':
                handled.add(msg)
        if self.position < len(self.list_objects):
            msgs_to_childs.append(self.list_objects[self.position])
            self.position += 1
        return (handled, MyList(msgs_to_childs), self.position < len(self.list_objects)) #return msgs set that where handled and new Messages
        
test = TestMessage()

summer = TestAdd()

test.childs = [summer]

l = test.process_msgs(MyList([('float', 2), ('float', 3), ('float', 7)]))

print(l)

t = testList()
t.list_objects = [('float', 1), ('float', 2), ('float', 3), ('float', 4)]

t.childs = [summer]
l = t.process_msgs(MyList([]))

print(l)


