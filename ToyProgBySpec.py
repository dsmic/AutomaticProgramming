#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:53:25 2020

@author: detlef
"""

from tkinter import Tk, Canvas, mainloop, W
master = Tk()

canvas_width = 800
canvas_height = 400
w = Canvas(master, 
           width=canvas_width,
           height=canvas_height)
w.pack()

def click(event):
    print('button clicked',event)


def key(event):
    c = w.create_text(20, 30, anchor=W, font=("Times New Roman", int(25), "bold"),
            text=event.char)
    print('key pressed',event, 'bounding', w.bbox(c))
    
    
w.bind('<Button-1>', click)
master.bind('<Key>',key)
mainloop()