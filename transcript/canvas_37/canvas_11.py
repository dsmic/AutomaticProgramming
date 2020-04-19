#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Graphics:

    def __init__(self, canvas_id):
        self.canvas_id = canvas_id
        self.canvas = document.getElementById(self.canvas_id)
        self.ctx = self.canvas.getContext('2d')
        #self.canvas.onmouseover = self.drawCanvas
        #self.canvas.onmouseup = self.set_active_false
        #self.canvas.onmousedown = self.set_active_true
        self.x, self.y = 0, 0 
        # default color
        self.ctx.fillStyle = "rgb(0, 255, 0)"
        #self.canvas.addEventListener('pointermove', self.drawCanvas)
        #self.canvas.addEventListener('pointerdown', self.mousedown)
        #self.canvas.addEventListener('pointerup', self.mouseup)
        self.canvas.addEventListener('touchmove', self.drawCanvas)
        self.canvas.addEventListener('touchstart', self.mousedown)
        self.canvas.addEventListener('touchend', self.mouseup)
        self.canvas.addEventListener('mousemove', self.drawCanvas)
        self.canvas.addEventListener('mousedown', self.mousedown)
        self.canvas.addEventListener('mouseup', self.mouseup)
        console.log('--------------------------------------------------------------------')

    def drawCanvas(self, event):
        # draw
        self.x = event.pageX-self.canvas.offsetLeft
        self.y = event.pageY-self.canvas.offsetTop
        # print (self.x, self.y)
        if self.active:
#            self.ctx.fillRect(self.x, self.y, 10, 10)
            eee = point(self.x, self.y)
            console.log("Hello world!")
            mousemove(eee)
            event.preventDefault()
            
    def mousedown(self, event):
        self.x = event.pageX-self.canvas.offsetLeft
        self.y = event.pageY-self.canvas.offsetTop
        if self.x < 30 and self.y < 30:
            mouseright(None)
            return
        self.set_active_true()
        event.preventDefault()
        
    def mouseup(self, event):
        self.x = event.pageX-self.canvas.offsetLeft
        self.y = event.pageY-self.canvas.offsetTop
        self.set_active_false()
        event.preventDefault()
        
    def set_active_true(self):
        # print ("true")
        if not self.active:
            self.active = True
            event = point(self.x, self.y)
            mousepress(event)

    def set_active_false(self):
        # print ("false")
        if self.active:
            self.active = False
            event = point(self.x, self.y)
            mouserelease(event)
    
    def set_fillStyle(self, color):
        if color == 'black':
            self.ctx.fillStyle = "rgb(0, 0, 0)"
            self.ctx.strokeStyle = "rgb(0, 0, 0)"
        elif color == 'red':
            self.ctx.fillStyle ="rgb(255, 0, 0)"
            self.ctx.strokeStyle ="rgb(255, 0, 0)"
        elif color == 'blue':
            self.ctx.fillStyle = "rgb(0, 0, 255)"
            self.ctx.strokeStyle = "rgb(0, 0, 255)"
        elif color == 'green':
            self.ctx.fillStyle = "rgb(0, 255, 0)"
            self.ctx.strokeStyle = "rgb(0, 255, 0)"
            
    def create_line(self, x1, y1, x2, y2, fill):
        self.set_fillStyle(fill)
        self.ctx.beginPath();
        self.ctx.moveTo(x1, y1);
        self.ctx.lineTo(x2, y2);
        self.ctx.stroke();
        
    def create_polygon(self, xx,  fill):
        self.set_fillStyle(fill)
        self.ctx.beginPath();
        self.ctx.moveTo(xx[0], xx[1]);
        for i in range(2, len(xx), 2):
            self.ctx.lineTo(xx[i], xx[i+1]);
        self.ctx.stroke();
        
    def create_oval(self, x1, y1, x2, y2, fill, outline):
        console.log(fill)
        if fill is not None:
            self.set_fillStyle(fill)
            self.ctx.beginPath()
            self.ctx.ellipse((x1+x2)/2, (y1+y2)/2, (x2-x1)/2, (x2-x1)/2, 0, 0, 2*Math.PI, True)
            self.ctx.fill()
            console.log('mark')
        if outline is not None:
            self.set_fillStyle(outline)
            self.ctx.beginPath()
            self.ctx.ellipse((x1+x2)/2, (y1+y2)/2, (x2-x1)/2, (x2-x1)/2, 0, 0, 2*Math.PI, True)
            self.ctx.stroke()
            console.log('mark')
        

w = None

def init():
    global w
    w = Graphics('graphics')
    clearCanvas()

def clearCanvas():
    w.ctx.clearRect(0, 0, w.canvas.width, w.canvas.height)
    w.ctx.fillStyle = "rgb(0, 255, 0)"
    w.ctx.fillRect(0, 0, 30, 30)
    w.ctx.strokeStyle = "rgb(255, 0, 0)"
    w.ctx.beginPath()
    w.ctx.moveTo(27, 3)
    w.ctx.lineTo(3,15)
    w.ctx.lineTo(27,27)
    w.ctx.stroke()
    


# pylint: disable=C0301, C0103, C0116, C0321, C0115, R0914, R0912, R0915, R1705, R1720, W0122, W0603, W0123, R1702, R0903, C0302

#import tkinter as tk
#from tkinter import Tk, Canvas, mainloop
import math
from math import sqrt


# Toy Manager
#master = Tk()
canvas_width = 2000
canvas_height = 1000
#w = Canvas(master, width=canvas_width, height=canvas_height)
#w.pack()

lastpress = None
lastpoints = None
last_line_properties = None

lastradius = None
clickedposition = None

lastdirect = 0

mark = None

def abst(a, b):
    print(a.x, a.y, b.x, b.y)
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2)

def kreuz(a, b, c):
    # sign of kreuzprodukt
    return Math.sign(1, (c.x-a.x)*(b.y-a.y)-(c.y-a.y)*(b.x-a.x))
def direct(a, b):
    return math.atan2(a.y-b.y, a.x-b.x)

def is_parallel(l1, l2):
    if l2 is None:
        return 0 # not parallel if nothing before
    return abs(abs(math.modf((l1-l2)/math.pi)[0])-0.5)*2

def mouserelease(event):
    global lastdirect, last_line_properties, lastpoints
    print('release', event)
    #******************************
    # this is working for draw the full line
    # pointflat = [a for l in lastpoints for a in (l.x,l.y)]
    # w.create_line(*pointflat, smooth=True, splinesteps=3)
    # print(pointflat)
    #******************************

    # calculate line properties
    start = lastpoints[0]
    end = lastpoints[len(lastpoints)-1]
    lp = lastpoints[0]
    ssum = 0
    for l in lastpoints[1:]:
        ssum += abst(l, lp)
        lp = l
    krum = 0
    for i in range(2, len(lastpoints)):
        c = abst(lastpoints[i-2], lastpoints[i])
        if c > 0:
            a = abst(lastpoints[i-1], lastpoints[i])
            b = abst(lastpoints[i-2], lastpoints[i-1])
            s = (a+b+c)/2
            try:
                print(a, b, c, s, 2/c*math.sqrt(s*(s-a)*(s-b)*(s-c)))
                krum += 2/c*math.sqrt(s*(s-a)*(s-b)*(s-c)) * kreuz(lastpoints[i-2], lastpoints[i-1], lastpoints[i])
                print(kreuz(lastpoints[i-2], lastpoints[i-1], lastpoints[i]), krum)
            except ValueError:
                print('ValueError')
    thisdirect = direct(start, end)
    ct = point((start.x+end.x)/2, (start.y+end.y)/2)
    if ssum > 0:
        kr = krum/ssum
    else:
        kr = 0
    line_properties = {'start': start, 'end': end, 'center': ct, 'length': ssum, 'curvature': kr, 'direction': thisdirect,
                       'parallel_to_last': is_parallel(thisdirect, lastdirect),
                       'pointlist': lastpoints}
    done = mcall(line_properties, last_line_properties)
    if done:
        last_line_properties = None
        lastdirect = None
        # w.delete("all")
        clearCanvas()
        lastpoints = None
    else:
        last_line_properties = line_properties #.copy()
        lastdirect = thisdirect
    for dd in draw_objects:
        print('draw', dd)
        dd.draw()
    if mark is not None:
        mark.draw()

def mousepress(event):
    global lastpress, lastpoints
    lastpress = event
    lastpoints = [event]
    console.log('lastpoints', lastpoints)
#    print('press', event)

def mousemove(event):
    global lastpoints
    # if abst(event,lastpoints[-1]) < 100:
    #     return
    lobj = len(lastpoints)-1
    console.log(lobj)
    w.create_line(lastpoints[lobj].x, lastpoints[lobj].y, event.x, event.y, 'black')
    lastpoints.append(event)
#    print('move', event)

class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class draw_point():
    def __init__(self, cx, cy, color):
        self.x = cx
        self.y = cy
        self.c = color

    def draw(self):
        cx = self.x
        cy = self.y
        w.create_line(cx-5, cy-5, cx+5, cy+5, self.c)
        w.create_line(cx-5, cy+5, cx+5, cy-5, self.c)
        print('indraw', cx, cy)

class draw_line():
    def __init__(self, sp, ep, sg=False, eg=False):
        self.sp = sp
        self.ep = ep
        self.sg = sg
        self.eg = eg
    def draw(self):
        sx = self.sp.x
        sy = self.sp.y
        ex = self.ep.x
        ey = self.ep.y
        # Halbgerade statt Strecke
        if self.eg:
            ex = ex + 1000*(ex-sx)
            ey = ey + 1000*(ey-sy)
        if self.sg:
            sx = sx + 1000*(sx-ex)
            sy = sy + 1000*(sy-ey)

        w.create_line(sx, sy, ex, ey, 'red')

class draw_polygon():
    def __init__(self, pg):
        self.pg = pg

    def draw(self):
        pg = self.pg
        if len(pg) >= 4:
#            w.create_line(*pg, fill="red")
            w.create_polygon(pg, 'red')

class draw_circle():
    def __init__(self, cp, radius):
        self.cp = cp
        self.radius = radius
    def draw(self):
        cp = self.cp
        radius = self.radius
        w.create_oval(cp.x - radius, cp.y - radius, cp.x +  radius, cp.y + radius, None, 'red')

class draw_mark():
    def __init__(self, cp, radius=3):
        self.x = cp.x
        self.y = cp.y
        self.radius = radius
    def draw(self):
        x = self.x
        y = self.y
        radius = self.radius
        console.log('try drawing')
        w.create_oval(x - radius, y - radius, x +  radius, y + radius, 'green', None)

class changed_line():
    def __init__(self, line, sg, eg):
        self.line = line
        self.sg = sg
        self.eg = eg
    def draw(self):
        pass
    def restore(self):
        self.line.sg = self.sg
        self.line.eg = self.eg

def find_point_near(ppoint, dist=None):
    mindist = None
    minpoint = None
    for p in draw_objects:
        if isinstance(p, draw_point):
            if mindist is None:
                mindist = abst(ppoint, p)
                minpoint = p
            else:
                ab = abst(ppoint, p)
                if ab < mindist:
                    mindist = ab
                    minpoint = p
    if mindist is None:
        return None
    if dist is None or mindist < dist:
        return minpoint
    return None

def find_line_near(ppoint, dist=None):
    minline = []
    for l in draw_objects:
        if isinstance(l, draw_line):
            for p in [l.sp, l.ep]:
                ab = abst(ppoint, p)
                if ab < dist:
                    minline.append(l)
    return minline

def check_strecke(xx, ll):
    sg = ll.sg
    eg = ll.eg
    if sg and eg:
        return True
    if sg:
        return xx <= 1
    if eg:
        #pylint: disable=C0122
        return 0 <= xx
    return 0 <= xx <= 1

def intersect_line_line(l1, l2):
    l1sx = l1.sp.x
    l1sy = l1.sp.y
    l1ex = l1.ep.x
    l1ey = l1.ep.y
    l2sx = l2.sp.x
    l2sy = l2.sp.y
    l2ex = l2.ep.x
    l2ey = l2.ep.y

    # create the solution coded below
    # g1 = 'l1sx + x1 * (l1ex-l1sx) - ( l2sx + x2 * (l2ex-l2sx))'
    # g2 = 'l1sy + x1 * (l1ey-l1sy) - ( l2sy + x2 * (l2ey-l2sy))'
    # r = sym.solve([g1,g2],['x1','x2'])

    try:
        x1 = (-(l1sx - l2sx)*(l2ey - l2sy) + (l1sy - l2sy)*(l2ex - l2sx))/((l1ex - l1sx)*(l2ey - l2sy) - (l1ey - l1sy)*(l2ex - l2sx))
        x2 = ((l1ex - l1sx)*(l1sy - l2sy) - (l1ey - l1sy)*(l1sx - l2sx))/((l1ex - l1sx)*(l2ey - l2sy) - (l1ey - l1sy)*(l2ex - l2sx))

        if check_strecke(x1, l1) and check_strecke(x2, l2):
            return [point(l1sx+ x1 * (l1ex-l1sx), l1sy+ x1 * (l1ey-l1sy))]
    except ZeroDivisionError as e:
        print('line_line', e)
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

    # create the solution coded below
    # g1 = '( l1sx + x1 * (l1ex-l1sx) - c2x )**2 + ( l1sy + x1 * (l1ey-l1sy) - c2y )**2 - radius **2'
    # r1 = sym.solve(g1,['x1'])

    try:
        x1 = (c2x*l1ex - c2x*l1sx + c2y*l1ey - c2y*l1sy - l1ex*l1sx - l1ey*l1sy + l1sx**2 + l1sy**2 - sqrt(-c2x**2*l1ey**2 + 2*c2x**2*l1ey*l1sy - c2x**2*l1sy**2 + 2*c2x*c2y*l1ex*l1ey - 2*c2x*c2y*l1ex*l1sy - 2*c2x*c2y*l1ey*l1sx + 2*c2x*c2y*l1sx*l1sy - 2*c2x*l1ex*l1ey*l1sy + 2*c2x*l1ex*l1sy**2 + 2*c2x*l1ey**2*l1sx - 2*c2x*l1ey*l1sx*l1sy - c2y**2*l1ex**2 + 2*c2y**2*l1ex*l1sx - c2y**2*l1sx**2 + 2*c2y*l1ex**2*l1sy - 2*c2y*l1ex*l1ey*l1sx - 2*c2y*l1ex*l1sx*l1sy + 2*c2y*l1ey*l1sx**2 - l1ex**2*l1sy**2 + l1ex**2*radius**2 + 2*l1ex*l1ey*l1sx*l1sy - 2*l1ex*l1sx*radius**2 - l1ey**2*l1sx**2 + l1ey**2*radius**2 - 2*l1ey*l1sy*radius**2 + l1sx**2*radius**2 + l1sy**2*radius**2))/(l1ex**2 - 2*l1ex*l1sx + l1ey**2 - 2*l1ey*l1sy + l1sx**2 + l1sy**2)

        if check_strecke(x1, l1):
            ret.append(point(l1sx + x1 * (l1ex-l1sx), l1sy + x1 * (l1ey-l1sy)))
    except (ValueError, ZeroDivisionError) as e:
        print('line_circle', e)
    try:
        x1 = (c2x*l1ex - c2x*l1sx + c2y*l1ey - c2y*l1sy - l1ex*l1sx - l1ey*l1sy + l1sx**2 + l1sy**2 + sqrt(-c2x**2*l1ey**2 + 2*c2x**2*l1ey*l1sy - c2x**2*l1sy**2 + 2*c2x*c2y*l1ex*l1ey - 2*c2x*c2y*l1ex*l1sy - 2*c2x*c2y*l1ey*l1sx + 2*c2x*c2y*l1sx*l1sy - 2*c2x*l1ex*l1ey*l1sy + 2*c2x*l1ex*l1sy**2 + 2*c2x*l1ey**2*l1sx - 2*c2x*l1ey*l1sx*l1sy - c2y**2*l1ex**2 + 2*c2y**2*l1ex*l1sx - c2y**2*l1sx**2 + 2*c2y*l1ex**2*l1sy - 2*c2y*l1ex*l1ey*l1sx - 2*c2y*l1ex*l1sx*l1sy + 2*c2y*l1ey*l1sx**2 - l1ex**2*l1sy**2 + l1ex**2*radius**2 + 2*l1ex*l1ey*l1sx*l1sy - 2*l1ex*l1sx*radius**2 - l1ey**2*l1sx**2 + l1ey**2*radius**2 - 2*l1ey*l1sy*radius**2 + l1sx**2*radius**2 + l1sy**2*radius**2))/(l1ex**2 - 2*l1ex*l1sx + l1ey**2 - 2*l1ey*l1sy + l1sx**2 + l1sy**2)

        if check_strecke(x1, l1):
            ret.append(point(l1sx + x1 * (l1ex-l1sx), l1sy + x1 * (l1ey-l1sy)))
    except (ValueError, ZeroDivisionError) as e:
        print('line_circle', e)
    return ret

def intersect_circle_circle(c1, c2):
    ret = []
    c1x = c1.cp.x
    c1y = c1.cp.y
    radius1 = c1.radius
    c2x = c2.cp.x
    c2y = c2.cp.y
    radius2 = c2.radius

    # create the solution coded below
    # g1 = '(  x  - c1x )**2 + ( y -   + c1y  )**2 - radius1**2'
    # g2 = '(  x  - c2x )**2 + ( y -   + c2y  )**2 - radius2**2'
    # r = sym.solve([g1,g2],['x','y'])
    # print(r)

    try:
        ev1 = (-(-c1x**2 - c1y**2 + c2x**2 + c2y**2 + radius1**2 - radius2**2 + (2*c1y - 2*c2y)*(-sqrt(-(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2 - radius1**2 - 2*radius1*radius2 - radius2**2)*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2 - radius1**2 + 2*radius1*radius2 - radius2**2))*(c1x - c2x)/(2*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2)) + (c1x**2*c1y + c1x**2*c2y - 2*c1x*c1y*c2x - 2*c1x*c2x*c2y + c1y**3 - c1y**2*c2y + c1y*c2x**2 - c1y*c2y**2 - c1y*radius1**2 + c1y*radius2**2 + c2x**2*c2y + c2y**3 + c2y*radius1**2 - c2y*radius2**2)/(2*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2))))/(2*(c1x - c2x)), -sqrt(-(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2 - radius1**2 - 2*radius1*radius2 - radius2**2)*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2 - radius1**2 + 2*radius1*radius2 - radius2**2))*(c1x - c2x)/(2*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2)) + (c1x**2*c1y + c1x**2*c2y - 2*c1x*c1y*c2x - 2*c1x*c2x*c2y + c1y**3 - c1y**2*c2y + c1y*c2x**2 - c1y*c2y**2 - c1y*radius1**2 + c1y*radius2**2 + c2x**2*c2y + c2y**3 + c2y*radius1**2 - c2y*radius2**2)/(2*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2)))
        ret.append(point(ev1[0], ev1[1]))
    except ValueError as e:
        print('no intersection circle_circle', e)

    try:
        ev2 = (-(-c1x**2 - c1y**2 + c2x**2 + c2y**2 + radius1**2 - radius2**2 + (2*c1y - 2*c2y)*(sqrt(-(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2 - radius1**2 - 2*radius1*radius2 - radius2**2)*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2 - radius1**2 + 2*radius1*radius2 - radius2**2))*(c1x - c2x)/(2*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2)) + (c1x**2*c1y + c1x**2*c2y - 2*c1x*c1y*c2x - 2*c1x*c2x*c2y + c1y**3 - c1y**2*c2y + c1y*c2x**2 - c1y*c2y**2 - c1y*radius1**2 + c1y*radius2**2 + c2x**2*c2y + c2y**3 + c2y*radius1**2 - c2y*radius2**2)/(2*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2))))/(2*(c1x - c2x)), sqrt(-(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2 - radius1**2 - 2*radius1*radius2 - radius2**2)*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2 - radius1**2 + 2*radius1*radius2 - radius2**2))*(c1x - c2x)/(2*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2)) + (c1x**2*c1y + c1x**2*c2y - 2*c1x*c1y*c2x - 2*c1x*c2x*c2y + c1y**3 - c1y**2*c2y + c1y*c2x**2 - c1y*c2y**2 - c1y*radius1**2 + c1y*radius2**2 + c2x**2*c2y + c2y**3 + c2y*radius1**2 - c2y*radius2**2)/(2*(c1x**2 - 2*c1x*c2x + c1y**2 - 2*c1y*c2y + c2x**2 + c2y**2)))
        ret.append(point(ev2[0], ev2[1]))
    except ValueError as e:
        print('no intersection circle_circle', e)

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
                    point_list.extend(intersect_line_line(o1, o2))
                if isinstance(o2, draw_circle):
                    point_list.extend(intersect_line_circle(o1, o2))
        if isinstance(o1, draw_circle):
            for o2 in tmp_draw:
                if isinstance(o2, draw_line):
                    point_list.extend(intersect_line_circle(o2, o1))
                if isinstance(o2, draw_circle):
                    point_list.extend(intersect_circle_circle(o1, o2))
    return point_list

draw_objects = []

def mouseright(_):
    global mark
    lo = draw_objects.pop()
    if isinstance(lo, changed_line):
        lo.restore()
    mark = None
#    w.delete("all")
    clearCanvas()
    for dd in draw_objects:
        print('draw', dd)
        dd.draw()

#w.bind('<ButtonRelease-1>', mouserelease)
#w.bind('<ButtonPress-1>', mousepress)
#w.bind('<ButtonPress-3>', mouseright)
#w.bind('<B1-Motion>', mousemove)

#master.bind('<Key>', key)


def mcall(lp, last_lp):
    # pylint: disable=R0911
    global mark, lastradius
    print('implemented', lp, last_lp)
    flatpoint = [p for s in lp['pointlist'] for p in (s.x, s.y)]

    # check for click
    if lp['length'] < 10:

        # set mark or click position
        if mark is None:
            np = find_point_near(lp['start'], 20)
            if np is None:
                intersecs = find_intersections()
                print('intersecs', intersecs)
                mindist = None
                minl = None
                for l in intersecs:
                    print('p', l.x, l.y)
                    dist = abst(lp['start'], l)
                    if mindist is None or dist < mindist:
                        mindist = dist
                        minl = l
                if mindist is not None and mindist < 30:
                    draw_objects.append(draw_point(minl.x, minl.y, 'blue'))
                    #mark = None
                else:
                    # this allows to draw points by just clicking, not making a cross
                    draw_objects.append(draw_point(lp['start'].x, lp['start'].y, 'red'))
            else:
                print('mark set')
                mark = draw_mark(np)
            return True

        # create a circle with the same radius
        if mark is not None:
            if abst(mark, lp['start']) < 30:
                draw_objects.append(draw_circle(mark, lastradius))

        mark = None
        return True


    # check for point was marked, now we make a circle from it
    if mark is not None:
        # check if point is near center
        pp = find_point_near(lp['center'], 20)
        if pp is None:
            pp = lp['center']
        radius = abst(mark, pp)
        lastradius = radius
        draw_objects.append(draw_circle(mark, radius))
        mark = None
        return True

    # check for a cross to mark points
    if last_lp is not None and 5 < last_lp['length'] < 80 and 5 < lp['length'] < 80 and lp['parallel_to_last'] < 0.3 and abst(lp['center'], last_lp['center']) < 20:
        cx = (last_lp['start'].x + last_lp['end'].x + lp['start'].x + lp['end'].x) / 4
        cy = (last_lp['start'].y + last_lp['end'].y + lp['start'].y + lp['end'].y) / 4
        draw_objects.pop()
        draw_objects.append(draw_point(cx, cy, 'red'))
        return True

    # check for a line between points
    sp = find_point_near(lp['start'], 20)
    ep = find_point_near(lp['end'], 20)
    if sp is not None and ep is not None:
        draw_objects.append(draw_line(sp, ep))
        return True

    # check if we do lengthen a line
    nl_list = find_line_near(lp['start'], 20)
    if len(nl_list) > 0:
        print('nl', nl_list)
        for nl in nl_list:
            # check for direction
            line_direct = direct(nl.sp, nl.ep)
            is_par = is_parallel(line_direct, lp['direction'])
            if is_par > 0.8:
                print('same direction')
                if abst(nl.sp, lp['start']) < 20:
                    if not nl.sg:
                        draw_objects.append(changed_line(nl, nl.sg, nl.eg))
                        nl.sg = True
                if abst(nl.ep, lp['start']) < 20:
                    if not nl.eg:
                        draw_objects.append(changed_line(nl, nl.sg, nl.eg))
                        nl.eg = True

        return True


    # The hand drawn element is saved
    draw_objects.append(draw_polygon(flatpoint))
    return False

#mainloop()


init()