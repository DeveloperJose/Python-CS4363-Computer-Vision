# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 2 HOG
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image
import numpy as np
from scipy import signal
import pdb
from pylab import *

convolution_matrix = np.array([[-1, 0, 1]]) 

def integral_sums_image(im, region, number_of_bars=12):
    # Initialize using the first color channel
    imx = np.zeros(im[:,:,0].shape)
    imx = signal.convolve(im[:,:,0], convolution_matrix, mode='same')
        
    imy = np.zeros(im[:,:,0].shape)
    imy = signal.convolve(im[:,:,0], np.transpose(convolution_matrix), mode='same')
    
    magnitude = np.sqrt(imx**2+imy**2)
    direction = (np.arctan2(imy, imx) * 180 / np.pi) + 180 # From 0 to 360    
    
    # Go through the other color channels
    # Get the maximum magnitude for each
    for i in range(1, 3):    
        imx = np.zeros(im[:,:,i].shape)
        imx = signal.convolve(im[:,:,i], convolution_matrix, mode='same')
        
        imy = np.zeros(im[:,:,i].shape)
        imy = signal.convolve(im[:,:,i], np.transpose(convolution_matrix), mode='same')
           
        curr_magnitude = np.sqrt(imx**2+imy**2)
        curr_direction = (np.arctan2(imy, imx) * 180 / np.pi) + 180 # From 0 to 360
        
        magnitude = np.maximum(magnitude, curr_magnitude)
        direction = np.maximum(direction, curr_direction)
    
    print "[Integral HOG] Finished getting magnitude and direction"
    
    start_time = timer()
    angle_step = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_step)
    
    (w, h) = region.shape[:2]
    (row, column) = im.shape[:2]
    row += 1
    column += 1
    integral_sums = np.zeros((12, row-h, column-w ), dtype=np.float64) 
    
    
    #integral_sums = np.zeros(12)
    i = 0
    print "[Integral HOG] Starting calculation of sums"
    for angle in angle_range:
        integral = magnitude.copy().astype(np.float64)
        # Set the magnitude for all other angles to 0
        integral[np.logical_or(direction<angle, direction>=angle+30)] = 0
        integral = integral.cumsum(1).cumsum(0)
        integral = np.pad(integral, (1, 0), 'constant', constant_values=(0))
        
        A = integral[0:row-h, 0:column-w]
        B = integral[0:row-h, w:column]
        C = integral[h:row, 0:column-w]
        D = integral[h:, w:]
        
        #pdb.set_trace()
        integral_sums[i] = A + D - B - C
        i = i + 1
        
    end_time = timer()    
    print('[Integral HOG] Duration: ' + str(end_time - start_time)) 

    return integral_sums

def integral_sums_region(im, number_of_bars=12):
    # Initialize using the first color channel
    imx = np.zeros(im[:,:,0].shape)
    imx = signal.convolve(im[:,:,0], convolution_matrix, mode='same')
        
    imy = np.zeros(im[:,:,0].shape)
    imy = signal.convolve(im[:,:,0], np.transpose(convolution_matrix), mode='same')
    
    magnitude = np.sqrt(imx**2+imy**2)
    direction = (np.arctan2(imy, imx) * 180 / np.pi) + 180 # From 0 to 360    
    
    # Go through the other color channels
    # Get the maximum magnitude for each
    for i in range(1, 3):    
        imx = np.zeros(im[:,:,i].shape)
        imx = signal.convolve(im[:,:,i], convolution_matrix, mode='same')
        
        imy = np.zeros(im[:,:,i].shape)
        imy = signal.convolve(im[:,:,i], np.transpose(convolution_matrix), mode='same')
           
        curr_magnitude = np.sqrt(imx**2+imy**2)
        curr_direction = (np.arctan2(imy, imx) * 180 / np.pi) + 180 # From 0 to 360
        
        magnitude = np.maximum(magnitude, curr_magnitude)
        direction = np.maximum(direction, curr_direction)
    
    print "[Region HOG] Finished getting magnitude and direction"
    
    start_time = timer()
    angle_step = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_step)
    
    integral_sums = np.zeros(12) 
    
    i = 0
    print "[Region HOG] Starting calculation of sums"
    for angle in angle_range:
        integral = magnitude.copy()
        # Set the magnitude for all other angles to 0
        integral[np.logical_or(direction<angle, direction>=angle+30)] = 0
        # Store the sum
        integral_sums[i] = integral.sum()
        i = i + 1
        
    end_time = timer()    
    print('[Region HOG] Duration: ' + str(end_time - start_time)) 

    return integral_sums
    
#interest_filename = "quijote1.jpg"
#im_file = Image.open(interest_filename)
#im = np.array(im_file)

#top_left = np.array([2064, 918])
#bottom_right = np.array([2552, 1674])

#region_x = top_left[0]
#region_y = top_left[1]
#region_width = bottom_right[1] - top_left[1]
#region_height = bottom_right[0] - top_left[0]
#region_image = im[region_y:region_y+region_width, region_x:region_x+region_height]

#image_integral_sums = integral_sums_image(im, region_image)
#region_sums = np.zeros(12)
#region_sums[:] = image_integral_sums[:, region_y+region_width, region_x+region_height]

# 354828.93161187653
#regular_sums = np.zeros(12)
#regular_sums = integral_sums_region(region_image)

# arr = np.array([[1,7,9, 5, 6], [3,11,1,5,2], [4,10,7,8,3], [8, 2, 0, 9, 5]])