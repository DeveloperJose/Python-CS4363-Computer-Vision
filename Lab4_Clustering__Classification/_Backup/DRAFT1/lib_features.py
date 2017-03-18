# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 4
Instructor: Olac Fuentes
"""
import numpy as np
from scipy import signal

# =========================================================================================
# ============================== Pixel Intensity
def pixel_intensity(I):
    if len(I.shape) == 3: # MNIST
        (total, w, h) = I.shape
        return np.reshape(I, (total, w * h))
    elif len(I.shape) == 4: # CIFAR
        (total, w, h, c) = I.shape
        return np.reshape(I, (total, w * h * c))
        
    raise Exception("[pixel_intensity] Invalid image dimensions: ", I.shape)

# =========================================================================================
# ============================== Histogram of Gradients
CONV_MATRIX = np.array([[-1, 0, 1]]) 

def gradient_histogram(I, number_of_bars):
    histograms = np.zeros((I.shape[0], number_of_bars))
    
    for index in range(I.shape[0]):
            if len(I.shape) == 4: # CIFAR
                histograms[index] = regular_color_gradient_histogram(I[index], number_of_bars)
            elif len(I.shape) == 3: # MNIST
                histograms[index] = regular_grayscale_gradient_histogram(I[index], number_of_bars)
            else:
                raise Exception("[gradient_histogram] Invalid image dimensions: ", I.shape)
                
    return histograms

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
    
    sums = np.zeros((number_of_bars), dtype=np.float64) 
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
# =========================================================================================
# ============================== Color Histogram
def color_histogram(I, number_of_bars):        
    if len(I.shape) == 4: # CIFAR Only
        (total, w, h, c) = I.shape
        histograms = np.zeros((total, 3, number_of_bars+1))
        for index in range(total):
            histograms[index] = regular_color_histogram(I[index], number_of_bars)

        return histograms.reshape(total, 3 * (number_of_bars+1))
        
    raise Exception("[color_histogram] Invalid image dimensions: ", I.shape) 

def regular_color_histogram(I, number_of_bars):
    hist = np.zeros((3, number_of_bars+1))
    Iq = np.uint8(I * float(number_of_bars) / 256.0)
    
    for bucket in range(number_of_bars):
        hist[0, bucket] = I[Iq[:,:,0] == bucket].sum()
        hist[1, bucket] = I[Iq[:,:,1] == bucket].sum()
        hist[2, bucket] = I[Iq[:,:,2] == bucket].sum()

    return hist
 
# =========================================================================================
# ============================== Daisy Features
from skimage.feature import daisy
from skimage import color

def daisy_feature(I):
    if len(I.shape) == 4: # CIFAR
        (total, w, h, c) = I.shape
        RADIUS = 15
        
    elif len(I.shape) == 3: # MNIST
        (total, w, h) = I.shape
        RADIUS=1
    
    else:
        raise Exception("[daisy_feature] Invalid image dimensions: ", I.shape) 
    
    # Calculate the first vector to find out the dimensions
    first_daisy = daisy(color.rgb2gray(I[0]), radius=RADIUS)
    (p, q, r) = first_daisy.shape
    
    # Create a place to store the features
    daisy_vector = np.zeros((total, p, q, r))
    # Add the already calculated feature
    daisy_vector[0] = first_daisy 
    
    for index in range(1, total):
        daisy_vector[index] = daisy(color.rgb2gray(I[index]), radius=RADIUS)

    (total, p, q, r) = daisy_vector.shape
    return np.reshape(daisy_vector, (total, p * q * r))