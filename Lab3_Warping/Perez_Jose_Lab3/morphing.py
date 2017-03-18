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

frames = 30
I = np.array(Image.open("face1.jpg"))
        
def cross_d(I, J, n):
    result = np.zeros((I.shape, n))
    result[:,:,:,0] = I
    result[:,:,:,n] = J

    for index in range(1, n):
        r_weight = (1/n) * index
        l_weight = 1 - r_weight
        result[:,:,:,index] = ((I * l_weight) + (J * r_weight))
        
    return result