# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 4
Instructor: Olac Fuentes
"""
import numpy as np
from scipy import signal
from skimage.feature import daisy
from skimage import color
# conda install -c https://conda.binstar.org/menpo opencv
import cv2

# =========================================================================================
# ============================== Pixel Intensity
def pixel_intensity(I, perform_deskew=False):
    if len(I.shape) == 3: # MNIST
        (total, w, h) = I.shape
        
        if perform_deskew:
            d1 = deskew(I[0])
            (w, h) = d1.shape
            temp = np.zeros((I.shape[0], w, h))
            for index in range(1, I.shape[0]):
                temp[index] = deskew(I[index])
            I = temp
    
        return np.reshape(I, (total, w * h))
    elif len(I.shape) == 4: # CIFAR
        (total, w, h, c) = I.shape
        return np.reshape(I, (total, w * h * c))
        
    raise Exception("[pixel_intensity] Invalid image dimensions: ", I.shape)

SZ = 20
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
    
# =========================================================================================
# ============================== Histogram of Gradients
def gradient_histogram(I, number_of_bars):
    histograms = np.zeros((I.shape[0], number_of_bars * 4))
    
    for index in range(I.shape[0]):
        histograms[index] = cv_hog(I[index], number_of_bars)
            #if len(I.shape) == 4: # CIFAR
                #histograms[index] = regular_color_gradient_histogram(I[index], number_of_bars)
            #elif len(I.shape) == 3: # MNIST
                #histograms[index] = regular_grayscale_gradient_histogram(I[index], number_of_bars)
            #else:
                #raise Exception("[gradient_histogram] Invalid image dimensions: ", I.shape)
                
    return histograms
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
# Histogram of Gradients using Sobel
# Divides region into 4 subregions
def cv_hog(img, number_of_bars):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(number_of_bars*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]

    hists = [np.bincount(b.ravel(), m.ravel(), number_of_bars) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist        
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