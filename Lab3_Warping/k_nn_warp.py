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

def k_nn_warp (I, source, dest):
    return I
    
I = np.array(Image.open("face2.jpg"))
imshow(I)

# Will take control points until a mouse middle click
points = np.asarray(ginput(0))

k = 3
# Image corners have no displacement
dest_t = np.array([[0,0],[I.shape[0],0],[0,I.shape[1]],[I.shape[0],I.shape[1]]])
disp_t = np.array([[0,0],[0,0],[0,0],[0,0]])

for i in range(len(points) / 2):
    p = points[i]
    p2 = points[i+1]
    disp = p2 - p
    
    dest_t = np.append(dest_t, [p])
    disp_t = np.append(disp_t, [disp])
    #table = np.append(table, [p2, p2 - p])
    
for r in range(I.shape[0]):
    for c in range(I.shape[1]):
        point = (r, c)
        dist_2 = np.zeros(dest_t.shape)
        for i in range(dest_t.shape[0]):
            dist_2[i] = np.sum((dest_t[i] - (r, c))**2)
        nearest = np.argsort(dist_2)[:k]
        #distances = disp_t[nearest]