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
    
    # Initialize using the first color channel
    imx = np.zeros(im[:,:,0].shape)
    imx = signal.convolve(im[:,:,0], convolution_matrix, mode='same')
        
    imy = np.zeros(im[:,:,0].shape)
    imy = signal.convolve(im[:,:,0], transpose(convolution_matrix), mode='same')
    
    magnitude = sqrt(imx**2+imy**2)
    direction = (np.arctan2(imy, imx) * 180 / pi) + 180 # From 0 to 360    
    
    # Go through the other color channels
    # Get the maximum magnitude for each
    for i in range(1, 3):    
        imx = np.zeros(im[:,:,i].shape)
        imx = signal.convolve(im[:,:,i], convolution_matrix, mode='same')
        
        imy = np.zeros(im[:,:,i].shape)
        imy = signal.convolve(im[:,:,i], transpose(convolution_matrix), mode='same')
           
        curr_magnitude = sqrt(imx**2+imy**2)
        curr_direction = (np.arctan2(imy, imx) * 180 / pi) + 180 # From 0 to 360
        
        magnitude = np.maximum(magnitude, curr_magnitude)
        direction = np.maximum(direction, curr_direction)
    
    start_time = timer()
    angle_step = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_step)
    
    (row, column) = imx.shape
    (height, width) = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    integral_sums = np.zeros((12, row-width, column-height)) 
    
    i = 0
    for angle in angle_range:
        integral = magnitude.copy()
        # Set the magnitude for all other angles to 0
        integral[logical_or(direction<angle, direction>=angle+30)] = 0
        # Create the integral image
        integral = integral.cumsum(axis=0).cumsum(axis=1)
        # Calculate A, B, C, and D
        A = integral[width:,height:]
        B = integral[width:row,:column-height]
        C = integral[:row-width,height:column]
        D = integral[:row-width,:column-height]
        # Store the sum
        integral_sums[i] = A + D - B - C
        i = i + 1
        
    end_time = timer()    
    print('[Integral HOG I Precalculation] Duration: ' + str(end_time - start_time)) 

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

image_hog = integral_sums_image(im, top_left, bottom_right)