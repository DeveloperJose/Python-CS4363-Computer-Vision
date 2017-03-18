# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 3
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
import numpy as np
from pylab import *

from real_index import real_index

im_file = Image.open("wall.jpg")
figure(1)
imshow(im_file)

im = array(im_file, dtype=np.float32)
corners = np.asarray(ginput(n=4), dtype=np.float32)
print "Selected corners"

(p0x, p0y) = corners[0]
(p1x, p1y) = corners[1]
(p2x, p2y) = corners[2]
(p3x, p3y) = corners[3]

p0 = array([p0y, p0x], dtype=np.int32)
p1 = array([p1y, p1x], dtype=np.int32)
p2 = array([p2y, p2x], dtype=np.int32)
p3 = array([p3y, p3x], dtype=np.int32)

width = 650
height = 1000
canvas_rounding = np.zeros((height, width, 3), dtype=np.uint8)
canvas_real = np.zeros((height, width, 3), dtype=np.uint8)

m = height
n = width
print "Starting loop"
start_time = timer()

for r in range(0, m):
    start = (r/(m-1.0)) * p3 + ((m-1.0-r)/(m-1.0)) * p0
    end = (r/(m-1.0)) * p2 + ((m-1.0-r)/(m-1.0)) * p1
    
    for c in range(0, n):
        p = (c/(n-1.0)) * end + ((n-1.0-c)/(n-1.0)) * start
        
        canvas_rounding[r, c, :] = im[int(p[0]+.5), int(p[1]+.5), :]
        canvas_real[r, c, :] = real_index(im, p)        

end_time = timer()                
print "Ending loop, duration: ", end_time - start_time

figure(2)
imshow(canvas_rounding) 
figure(3)
imshow(canvas_real)
show()