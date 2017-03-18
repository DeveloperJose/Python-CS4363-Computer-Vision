# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 1
Instructor: Olac Fuentes
Last Modification: September 2, 2016 by Jose Perez
"""
from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
from numpy import *
from pylab import *

def problem1(filename):
    gray()
    im = array(Image.open(filename).convert('L'))
    figure(0)
    imshow(im)
    for i in range(5):
        diff = abs(im - filters.gaussian_filter(im, 3))
    
        figure(i + 1)    
        imshow(diff)
        im = diff

def problem2(filename):
    gray()
    original = Image.open(filename).convert('L')
    
    scale_width = 50
    scale_height = 50
    scale = (scale_width, scale_height)
    
    figure(0)
    imshow(original)
    
    im = original
    for i in range(5):
        im_array = array(im)
        im_scale = im.resize(scale).resize(im_array.shape)
        diff = abs(im_array - array(im_scale))    
        im = Image.fromarray(diff)
        figure(i + 1)
        imshow(diff)
    
def problem3(filename, top_left, bottom_left, number_of_bars):
    # Calculate the rectangle to crop the image    
    distance = bottom_left - top_left
    crop_rect = append(top_left, distance)
    
    # Open the file and crop it
    im_file = Image.open(filename).convert('L')
    im = array(im_file.crop(crop_rect))
    
    start_time = timer()      
    
    # Convolution
    # Calculate magnitude and direction
    # Direction will be in degrees
    from scipy.ndimage.filters import convolve
    convolution_matrix =  array([[-1, 0, 1], [-1,0,1], [-1, 0, 1]])   
    
    imx = zeros(im.shape)
    imx = convolve(im, convolution_matrix)
      
    imy = zeros(im.shape)
    imy = convolve(im, transpose(convolution_matrix))   
       
    magnitude = sqrt(imx**2+imy**2)
    direction = arctan2(imy, imx) * 180 / pi
  
    # Put angles together depending on the number of bars  
    angle_distance = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_distance)
    bins = []
    for angle in angle_range:
        current_bin = logical_and(direction>=angle, direction<angle+30)    
        bins.append(current_bin)
    
    # Sum the magnitudes of the angle bins    
    magnitude_sums = []
    
    integral_image = im.cumsum(axis=0).cumsum(axis=1)
    
    for i in range(len(bins)):    
        magnitude_for_bin = magnitude[bins[i]]
        total_magnitude = magnitude_for_bin.sum()
        magnitude_sums.append(total_magnitude)        
    
    end_time = timer()    
    print('[Problem 3] Duration: ' + str(end_time - start_time))    
      
    # Show the cropped figure
    figure(1)
    imshow(im)    
    
    # Prepare the bar graph
    fig = figure(2)
    splot = fig.add_subplot(111)
    
    splot.bar(range(len(magnitude_sums)), magnitude_sums, width=1)
    
    xTickMarks = [str(i) + "-"+ str(i + angle_distance) for i in angle_range]
    splot.set_xticklabels(xTickMarks)
    splot.set_ylabel("Magnitude")
    splot.set_xlabel("Angles")
    show()
    
def problem4(filename, top_left, bottom_left, number_of_bars):
    # Calculate the rectangle to crop the image    
    distance = bottom_left - top_left
    crop_rect = append(top_left, distance)
    
    # Open the file and crop it
    im_file = Image.open(filename).convert('L')
    im = array(im_file.crop(crop_rect))
    
    start_time = timer()      
    
    # Convolution
    # Calculate magnitude and direction
    # Direction will be in degrees
    from scipy.ndimage.filters import convolve
    convolution_matrix =  array([[-1, 0, 1], [-1,0,1], [-1, 0, 1]])   
    
    imx = zeros(im.shape)
    imx = convolve(im, convolution_matrix)
      
    imy = zeros(im.shape)
    imy = convolve(im, transpose(convolution_matrix))   
       
    magnitude = sqrt(imx**2+imy**2)
    direction = arctan2(imy, imx) * 180 / pi
  
    # Put angles together depending on the number of bars  
    angle_distance = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_distance)
    bins = []
    for angle in angle_range:
        current_bin = logical_and(direction>=angle, direction<angle+30)    
        bins.append(current_bin)
    
    # Sum the magnitudes of the angle bins    
    magnitude_sums = []
    integral_image = im.cumsum(axis=0).cumsum(axis=1)
    
    for i in range(len(bins)):    
        magnitude_for_bin = magnitude[bins[i]]
        
        if len(integral_image) > 0:
            total_magnitude = integral_image[len(integral_image)-1]
            magnitude_sums.append(total_magnitude)
        else:
            magnitude_sums.append(0)
    
    end_time = timer()    
    print('[Problem 4] Duration: ' + str(end_time - start_time))    
     
    # Show the cropped figure
    figure(1)
    imshow(im)    
    
    # Prepare the bar graph
    fig = figure(2)
    splot = fig.add_subplot(111)
    
    splot.bar(range(len(magnitude_sums)), magnitude_sums, width=1)
    #xTickMarks = [str(i) + "-" + str(i + 30) for i in xrange(0, 360, 30)]    
    
    xTickMarks = [str(i) + "-" + str(i + angle_distance) for i in angle_range]
    splot.set_xticklabels(xTickMarks)
    splot.set_ylabel("Magnitude")
    splot.set_xlabel("Angles")
    show()
    
def demo():
    filename = 'img_large.jpg'
    top_left = array([0, 0])
    bottom_left = array([1000, 1000])
    number_of_bars = 12
    
    problem3(filename, top_left, bottom_left, number_of_bars)
    problem4(filename, top_left, bottom_left, number_of_bars)