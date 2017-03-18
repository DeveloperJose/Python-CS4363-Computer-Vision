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

# For 1200x2000
# No Loop: 38.12s
# Loop: 49.54s

# For 400x1000
# No Loop: 6.40s
# Loop: 8.25s
width = 650
height = 1000
canvas_rounding = np.zeros((height, width, 3), dtype=np.uint8)
canvas_real_loop = np.zeros((height, width, 3), dtype=np.uint8)
canvas_real_no_loop = np.zeros((height, width, 3), dtype=np.uint8)

m = height
n = width
for r in range(0, m):
    start = (r/(m-1.0)) * p3 + ((m-1.0-r)/(m-1.0)) * p0
    end = (r/(m-1.0)) * p2 + ((m-1.0-r)/(m-1.0)) * p1
    
    for c in range(0, n):
        p = (c/(n-1.0)) * end + ((n-1.0-c)/(n-1.0)) * start
        
        canvas_rounding[r, c, :] = im[int(p[0]), int(p[1]), :]
        
        colorsLoop, colorsNoLoop = realIndex(im, p)
        canvas_real_loop[r, c, :] = colorsLoop
        canvas_real_no_loop[r, c, :] = colorsNoLoop         

canvas_backward = np.zeros((height,width,3), dtype=np.uint8)

print "Total Time (No Loop)", (noLoopDuration)
print "Total Time (Loop)", (loopDuration)
                
figure(2)
imshow(canvas_rounding) 
figure(3)
imshow(canvas_real_loop)
figure(4)
imshow(canvas_real_no_loop) 
 
show()