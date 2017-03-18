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

(p0x, p0y) = corners[0]
(p1x, p1y) = corners[1]
(p2x, p2y) = corners[2]
(p3x, p3y) = corners[3]

p0 = array([p0y, p0x], dtype=np.int32)
p1 = array([p1y, p1x], dtype=np.int32)
p2 = array([p2y, p2x], dtype=np.int32)
p3 = array([p3y, p3x], dtype=np.int32)

im_insert_file = Image.open("album.jpg")
im_insert = np.array(im_insert_file, dtype=np.uint8)
width = im_insert_file.width
height = im_insert_file.height

left = min(p0x, p1x, p2x, p3x)
right = max(p0x, p1x, p2x, p3x)

top = max(p0y, p1y, p2y, p3y)
bottom = min(p0y, p1y, p2y, p3y)

width_interp = right - left
height_interp = top - bottom
im_insert = np.array(im_insert_file.resize((width_interp, height_interp)))
canvas = np.array(im_file.copy())
print "Starting loop"
start_time = timer()
fig = figure(2)
imshow(canvas)
splot = fig.add_subplot(111)
import matplotlib.patches as patches

for w in range(0, width_interp):
    for h in range(0, height_interp):
        r0_y = p0y + h
        r0_x = np.interp(r0_y, np.array([p0y, p3y]), np.array([p0x, p3x]))
        #rect = patches.Rectangle((r0_x, r0_y), 3, 3, linewidth=1, edgecolor='r', facecolor='none')
        #splot.add_patch(rect)
        
        r1_y = p1y + h
        r1_x = np.interp(r1_y, np.array([p1y, p2y]), np.array([p1x, p2x]))
        #rect = patches.Rectangle((r1_x, r1_y), 3, 3, linewidth=1, edgecolor='r', facecolor='none')
        #splot.add_patch(rect)
        
        x = r0_x + w
        y = np.interp(x, np.array([r0_x, r1_x]), np.array([r0_y, r1_y]))
        rect = patches.Rectangle((x, y), 3, 3, linewidth=1, edgecolor='r', facecolor='none')
        splot.add_patch(rect)
        # Get the color from the rectangular object
        colors = real_index(im_insert, np.array([h, w]))
        # Place it on the interpolated position
        canvas[(int)(y),(int)(x), :] = colors
        
end_time = timer()                
print "Ending loop, duration: ", end_time - start_time    
figure(3)
imshow(canvas)
show()