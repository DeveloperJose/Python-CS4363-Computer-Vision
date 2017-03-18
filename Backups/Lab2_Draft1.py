# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 2
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image
import numpy as np
import cPickle as pickle
from scipy import signal
from pylab import *

def integral_sums_image(im, top_left, bottom_right, number_of_bars=12):
    convolution_matrix = array([[-1, 0, 1]]) 
    imx = np.zeros(im.shape)
    imx = signal.convolve(im, convolution_matrix, mode='same')
    
    imy = np.zeros(im.shape)
    imy = signal.convolve(im, transpose(convolution_matrix), mode='same')
       
    magnitude = sqrt(imx**2+imy**2)
    direction = (np.arctan2(imy, imx) * 180 / pi) + 180 # From 0 to 360
    
    start_time = timer()
    angle_step = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_step)
    
    (row, column) = im.shape
    (height, width) = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    integral_sums = np.zeros((12, row-width, column-height)) 
    #test = np.zeros(12)
    i = 0
    for angle in angle_range:
        integral = magnitude.copy()
        integral[logical_or(direction<angle, direction>=angle+30)] = 0
        integral = integral.cumsum(axis=0).cumsum(axis=1)
        A = integral[width:,height:]
        B = integral[width:row,:column-height]
        C = integral[:row-width,height:column]
        D = integral[:row-width,:column-height]
        Sum = A + D - B - C
        integral_sums[i] = Sum
    
    end_time = timer()    
    print('[Integral HOG I Precalculation] Duration: ' + str(end_time - start_time)) 

    return integral_sums

def integral_sums_region(region, number_of_bars=12):
    convolution_matrix = array([[-1, 0, 1]]) 
    
    imx = np.zeros(region.shape)
    imx = signal.convolve(region, convolution_matrix, mode='same')
    
    imy = np.zeros(region.shape)
    imy = signal.convolve(region, transpose(convolution_matrix), mode='same')
       
    magnitude = sqrt(imx**2+imy**2)
    direction = (np.arctan2(imy, imx) * 180 / pi) + 180 # From 0 to 360
    
    start_time = timer()
    angle_step = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_step)
    
    integral_sums = np.zeros(12)    
    i = 0
    for angle in angle_range:
        integral = magnitude.copy()
        integral[logical_or(direction<angle, direction>=angle+30)] = 0
        integral = integral.cumsum(axis=0).cumsum(axis=1)
        
        integral_sums[i] = integral[region.shape[0]-1, region.shape[1]-1]
        
        i = i + 1
    
    end_time = timer()    
    print('[Integral HOG R Precalculation] Duration: ' + str(end_time - start_time))
    
    return integral_sums

im_file = Image.open("quijote1.jpg")
im = array(im_file)

# Selected region of interest
top_left = array([2064, 918])
bottom_right = array([2552, 1674])

# Region Variables
region_x = top_left[0]
region_y = top_left[1]
region_width = bottom_right[1] - top_left[1]
region_height = bottom_right[0] - top_left[0]
region_image = im[region_y:region_y+region_width, region_x:region_x+region_height]

# We have to pick a color channel
# Do so by picking the one with the highest cummulative magnitude
region_sums = integral_sums_region(region_image[:,:,2])
size = region_sums.sum()
#for i in range(2):
    #current_color_sums = integral_sums_region(region_image[:,:,i])
    #magnitude = current_color_sums.sum()
    # Pick the color with the highest cummulative magnitude
    #if magnitude > size:
        #size = magnitude
        #region_sums = current_color_sums

# Comparison
im_file_compare = Image.open("quijote1.jpg")
im_compare = array(im_file_compare)

# Calculate the HOG of the entire image
# Also pick the highest cummulative magnitude channel
compare_sums = integral_sums_image(im_compare[:,:,2], top_left, bottom_right)
compare_size = compare_sums.sum()

#for i in range(2):
    #current_color_sums = integral_sums_image(im_compare[:,:,i], top_left, bottom_right)
    #magnitude = current_color_sums.sum()
    # Pick the color with the highest cummulative magnitude
    #if magnitude > compare_size:
        #compare_size = magnitude
        #compare_sums = current_color_sums
        
# Find the best match
stride = 1
distance = sys.maxsize
best_match = (-1, -1)

print "Comparing..."
start_time = timer()

for x in range(0, compare_sums.shape[2]-1, stride):
    for y in range(0, compare_sums.shape[1]-1, stride):
        comp = compare_sums[:, y-1, x-1]
        
        if (region_sums.shape != comp.shape):
            continue
        
        diff = sqrt(((region_sums - comp)**2).sum())
        
        if diff < distance:
            distance = diff
            best_match = (y, x)
    
end_time = timer()
print "Matching took ", end_time - start_time

fig = figure(2)
imshow(im_compare)
splot = fig.add_subplot(111)
import matplotlib.patches as patches
rect = patches.Rectangle(best_match,region_height,region_width,linewidth=1,edgecolor='r',facecolor='none')
splot.add_patch(rect)
show()