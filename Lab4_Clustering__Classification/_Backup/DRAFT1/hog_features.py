# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:22:13 2016

@author: xeroj
"""


def regular_grayscale_gradient_histogram(I, number_of_bars):
    global CONV_MATRIX
    # Initialize using the first color channel
    imx = np.zeros(I.shape)
    imx = signal.convolve(I, CONV_MATRIX, mode='same')
        
    imy = np.zeros(I.shape)
    imy = signal.convolve(I, np.transpose(CONV_MATRIX), mode='same')
    
    magnitude = np.sqrt(imx**2+imy**2)
    direction = (np.arctan2(imy, imx) * 180 / np.pi) + 180 # From 0 to 360    
    
    angle_step = 360 / number_of_bars
    
    sums = np.zeros((number_of_bars+1), dtype=np.float64) 
    i = 0
    for angle in xrange(0, 360, angle_step):
        integral = magnitude.copy().astype(np.float64)
        sums[i] = integral[np.logical_and(direction > angle, direction <= angle + angle_step)].sum()
        i = i + 1
    
    return sums
    
def regular_color_gradient_histogram(I, number_of_bars):
    global CONV_MATRIX
    # Initialize using the first color channel
    imx = np.zeros(I[:,:,0].shape)
    imx = signal.convolve(I[:,:,0], CONV_MATRIX, mode='same')
        
    imy = np.zeros(I[:,:,0].shape)
    imy = signal.convolve(I[:,:,0], np.transpose(CONV_MATRIX), mode='same')
    
    magnitude = np.sqrt(imx**2+imy**2)
    direction = (np.arctan2(imy, imx) * 180 / np.pi) + 180 # From 0 to 360    
    
    # Go through the other color channels
    # Get the maximum magnitude for each
    for i in range(1, 3):    
        imx = np.zeros(I[:,:,i].shape)
        imx = signal.convolve(I[:,:,i], CONV_MATRIX, mode='same')
        
        imy = np.zeros(I[:,:,i].shape)
        imy = signal.convolve(I[:,:,i], np.transpose(CONV_MATRIX), mode='same')
           
        curr_magnitude = np.sqrt(imx**2+imy**2)
        curr_direction = (np.arctan2(imy, imx) * 180 / np.pi) + 180 # From 0 to 360
        
        magnitude = np.maximum(magnitude, curr_magnitude)
        direction = np.maximum(direction, curr_direction)
    
    angle_step = 360 / number_of_bars
    
    sums = np.zeros((number_of_bars), dtype=np.float64) 
    i = 0
    for angle in xrange(0, 360, angle_step):
        integral = magnitude.copy().astype(np.float64)
        sums[i] = integral[np.logical_and(direction > angle, direction <= angle + angle_step)].sum()
        i = i + 1
    
    return sums
