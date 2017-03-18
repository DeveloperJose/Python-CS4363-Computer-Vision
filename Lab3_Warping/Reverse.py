# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:25:42 2016

@author: jgpd
"""
# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Warping Quiz
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
import numpy as np
from pylab import *

noLoopDuration = 0 
noLoopCount = 0

loopDuration = 0
loopCount = 0

def realIndex(im, p):
    rp = p[0]
    cp = p[1]
    
    r = int(floor(rp))    
    c = int(floor(cp))
    
    dr2 = rp - r
    dr1 = 1 - dr2
    
    dc2 = cp - c
    dc1 = 1 - dc2
    
    dr = transpose(array([dr1, dr2]))
    dc = array([dc1, dc2])    
    
    weights = np.outer(dr, dc)
    
    start_time = timer()
    colorsNoLoop = np.sum(np.sum(im[r,c,:].reshape((3, 1, 1)) * weights
                    , axis=1),axis=1)   
    end_time = timer()
    
    global noLoopDuration
    noLoopDuration += end_time - start_time
    global noLoopCount
    noLoopCount += 1
    
    colorsLoop = np.zeros(3)
    start_time = timer()        
    for i in range(3):
        colorsLoop[i] = np.sum(im[r,c,i] * weights)
    
    end_time = timer()
    
    global loopDuration
    loopDuration += end_time - start_time
    global loopCount 
    loopCount += 1
    
    return colorsLoop, colorsNoLoop

im_file = Image.open("wall.jpg")
figure(1)
imshow(im_file)

im = array(im_file, dtype=np.float32)
corners = np.asarray(ginput(n=4), dtype=np.float32)

(p0x, p0y) = corners[0]
(p1x, p1y) = corners[1]
(p2x, p2y) = corners[2]
(p3x, p3y) = corners[3]

p0 = array([p0y, p0x], dtype=np.int32)
p1 = array([p1y, p1x], dtype=np.int32)
p2 = array([p2y, p2x], dtype=np.int32)
p3 = array([p3y, p3x], dtype=np.int32)

im_insert_file = Image.open("corrin.png")
im_insert = np.array(im_insert_file, dtype=np.uint8)
width = im_insert_file.width
height = im_insert_file.height

left = min(p0x, p1x, p2x, p3x)
right = max(p0x, p1x, p2x, p3x)

top = max(p0y, p1y, p2y, p3y)
bottom = min(p0y, p1y, p2y, p3y)

canvas = np.zeros((top - bottom, right - left, 3))
print "Begin"
for w in range(0, width - 1):
    for h in range(0, height - 1):
        r0_y = h
        r0_x = np.interp(r0_y, np.array([p0y, p3y]), np.array([p0x, p3x]))
        
        r1_y = h
        r1_x = np.interp(r1_y, np.array([p1y, p2y]), np.array([p1x, p2x]))
        
        x = r0_y + w
        y = np.interp(x, np.array([r0_x, r1_x]), np.array([r0_y, r1_y]))
        
        colorsLoop, colorsNoLoop = realIndex(im, np.array([x, y]))
        canvas[(int)(x),(int)(y), :] = colorsLoop
        
imshow(canvas) 
show()
