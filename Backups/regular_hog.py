# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 01:10:58 2016

@author: jgpd
"""

def regular_hog(im_file, top_left, bottom_right, number_of_bars=12):
    im = array(im_file, dtype=np.float32)
    x = top_left[0]
    y = top_left[1]
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]
        
    imx = zeros(im.shape)
    imx = signal.convolve(im, convolution_matrix)[y:y+width, x:x+height]
      
    imy = zeros(im.shape)
    imy = signal.convolve(im, transpose(convolution_matrix))[y:y+width, x:x+height]  
       
    magnitude = sqrt(imx**2+imy**2)
    # Direction will be from 0 to 360
    direction = (arctan2(imy, imx) * 180 / pi) + 180
    
    start_time = timer()
    angle_step = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_step)
    bins = []
    for angle in angle_range:
        current_bin = logical_and(direction>=angle, direction<angle+30)    
        bins.append(current_bin)
    
    # Sum the magnitudes of the angle bins    
    magnitude_sums = []
    
    for i in range(len(bins)):        
        magnitude_sums.append(magnitude.copy()[bins[i]].sum())          
    
    end_time = timer()    
    print('[Normal HOG] Duration: ' + str(end_time - start_time))     
    
    #figure(2)
    temp = magnitude.copy()
    temp[invert(bins[0])] = 0    
    #imshow(temp)    
    
    return magnitude_sums, temp