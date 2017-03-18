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

noLoopDuration = 0 
loopDuration = 0

def real_index(im, p):
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

    # Performs slightly faster
    #colorsNoLoop = np.sum(np.sum(im[r,c,:].reshape((3, 1, 1)) * weights
    #                , axis=1),axis=1)   
    
    colorsLoop = np.zeros(3)
    for i in range(3):
        colorsLoop[i] = np.sum(im[r,c,i] * weights)
    
    return colorsLoop