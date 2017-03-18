# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:17:03 2016

@author: jgpd
"""
from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
import numpy as np
from pylab import *
from scipy.ndimage.filters import convolve

im_file = Image.open("quijote1.jpg").convert("L")

top_left = array([2064, 893])
bottom_right = array([2608, 1706])

x = top_left[0]
y = top_left[1]
distance = bottom_right - top_left
im_array = array(im_file)[x:x+distance[0], y:y+distance[1]]
number_of_bars = 12

convolution_matrix =  array([[-1, 0, 1], [-1,0,1], [-1, 0, 1]])   

imx = zeros(im_array.shape)
imx = convolve(im_array, convolution_matrix)
  
imy = zeros(im_array.shape)
imy = convolve(im_array, transpose(convolution_matrix))   
   
magnitude = sqrt(imx**2+imy**2)
#direction = degrees(2 * arctan2(1, 1/imx)-imy)
direction = arctan2(imy, imx) * 180 / pi

# Put angles together depending on the number of bars  
angle_distance = 360 / number_of_bars
angle_range = xrange(0, 360, angle_distance)
bins = []
for angle in angle_range:
    current_bin = logical_and(direction>=angle, direction<angle+30)    
    #current_bin = logical_or(direction<angle, direction>=angle+30)        
    bins.append(current_bin)

integral_images = []    
for i in range(len(bins)):
    current_bin = bins[i]
    integral = magnitude.copy()
    integral[current_bin] = 0
    #integral.cumsum(axis=0).cumsum(axis=1)
    integral_images.append(integral)
    
width = 1
height = 1
(row, column) = im_array.shape

integral = integral_images[0]
A = integral[:width,:height]
B = integral[width:row,:column-height]
C = integral[:row-width,height:column]
D = integral[:row-width,:column-height]

Sum = A + D - B - C