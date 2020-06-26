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
    def __init__(self, parent):
        self.parent = parent
        self.childs = [] # this are the childs, which get the messages
        self.subobjects = [] # objects, for with the childs are handled
        pass
    
    def process_msgs(self, msgs):
        print(type(self), 'process_msgs', msgs)
        (msgs_all_handled, msgs_all_added) = self.msgs_receive(msgs)
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
        
        # call self.msgs_receive a second time? with msgs_all_added and leave the rest????
        (msgs_second_handled, msgs_second_added) = self.msgs_receive(msgs_to_childs + msgs_all_added)
        
        rest_added_msgs = MyList(msgs_all_added + msgs_second_added)
        msgs_all_handled.update(msgs_second_handled)
        rest_added_msgs.remove_set(msgs_second_handled)
              
        return (msgs_all_handled, rest_added_msgs)
        
    
    # overwrite
    # remove the messages from msg, which are handled (maybe also by the childs??), the rest is kept
    def msgs_receive(self, msgs):
        print(type(self), 'msgs_received', msgs)
        return (set(), MyList([])) #return msgs set that where handled and new Messages
    
    def msg_string():
        return None
    

class TestMessage(BaseClassMSG):
    def msgs_receive(self, msgs):
        print(type(self), 'msgs_received', msgs)
        return (set(), MyList([])) #return msgs set that where handled and new Messages

    def msg_string():
        return 'create_TestMessage'
        
class TestAdd(BaseClassMSG):
    def msgs_receive(self, msgs):
        print(type(self), 'msgs_received', msgs)
        summe = 0
        handled = set()
        for msg in msgs:
            (msg_type, value) = msg
            if msg_type == 'float':
                handled.add(msg)
                summe += value
        if len(handled) == 0:
            return (handled, [])
        return (handled, [('result', summe)])

    def msg_string():
        return 'create_TestAdd'

class testList(BaseClassMSG):
    def __init__(self, parent):
        super.__init__(self, parent)
        self.list_objects = []
    
    def msgs_receive(self, msgs):
        print(type(self), 'msgs_received', msgs)
        handled = set()
        msgs_to_childs = []
        for msg in msgs:
            (msg_type, value) = msg
            if msg_type == 'float':
                handled.add(msg)
                msgs_to_childs.append(msg)
                
                
                
        return (set(), MyList([])) #return msgs set that where handled and new Messages
        
test = TestMessage(None)

summer = TestAdd(test)

test.childs = [summer]

l = test.process_msgs(MyList([('float', 2), ('float', 3), ('float', 7)]))

print(l)


