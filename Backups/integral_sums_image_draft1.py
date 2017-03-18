# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from PIL import Image
import numpy as np
import cPickle as pickle
from scipy import signal
from pylab import *

number_of_bars = 12
im_file = Image.open("quijote1.jpg")
im = array(im_file)

# Selected region of interest
top_left = array([2064, 918])
bottom_right = array([2552, 1674])

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

(row, column) = im[:,:,0].shape
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
    i += 1
end_time = timer()    
print('[Integral HOG I Precalculation] Duration: ' + str(end_time - start_time)) 
region_sums[:] = integral_sums[:, region_y+region_width, region_x+region_height]

region_x = top_left[0]
region_y = top_left[1]
region_width = bottom_right[1] - top_left[1]
region_height = bottom_right[0] - top_left[0]
region_image = im[region_y:region_y+region_width, region_x:region_x+region_height]
region = region_image

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

integral_sums_2 = np.zeros(12)    
i = 0
# 376490.1341911073
for angle in angle_range:
    integral = magnitude.copy()
    integral[logical_or(direction<angle, direction>=angle+30)] = 0
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    
    integral_sums_2[i] = integral[region.shape[0]-1, region.shape[1]-1]
    
    i = i + 1

end_time = timer()    
print('[Integral HOG R Precalculation] Duration: ' + str(end_time - start_time))

