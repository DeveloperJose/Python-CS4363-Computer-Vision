# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 1
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
from numpy import *
from pylab import *

def get_hog_data(im_array, number_of_bars=12):
    from scipy.ndimage.filters import convolve
    convolution_matrix =  array([[-1, 0, 1], [-1,0,1], [-1, 0, 1]])   
    
    imx = zeros(im_array.shape)
    imx = convolve(im_array, convolution_matrix)
      
    imy = zeros(im_array.shape)
    imy = convolve(im_array, transpose(convolution_matrix))   
       
    magnitude = sqrt(imx**2+imy**2)
    direction = degrees( 2 * arctan2(1, 1/imx)-imy)

    # Put angles together depending on the number of bars  
    angle_distance = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_distance)
    bins = []
    for angle in angle_range:
        #current_bin = logical_and(direction>=angle, direction<angle+30)    
        current_bin = logical_and(direction<angle, direction>=angle+30)        
        bins.append(current_bin)

    integral_images = []    
    for i in range(len(bins)):
        current_bin = bins[i]
        integral = magnitude
        integral[current_bin] = 0
        integral_images.append(integral)
    
    return integral

#def hog(im_array, bins, magnitude):    
    #start_time = timer()            
    
    
    
    # Sum the magnitudes of the angle bins    
    #magnitude_sums = []
    
    #for i in range(len(bins)):    
    #    magnitude_for_bin = magnitude[bins[i]]
     #   total_magnitude = magnitude_for_bin.sum()
     #   magnitude_sums.append(total_magnitude)        
    
    #end_time = timer()    
    #print('[HOG] Duration: ' + str(end_time - start_time))    
    
    # Prepare the bar graph
    #fig = figure(2)
    #splot = fig.add_subplot(111)
    
    #splot.bar(range(len(magnitude_sums)), magnitude_sums, width=1)
    #xTickMarks = [str(i) + "-" + str(i + 30) for i in xrange(0, 360, 30)]    
    
    #xTickMarks = [str(i) + "-" + str(i + angle_distance) for i in angle_range]
    #splot.set_xticklabels(xTickMarks)
    #splot.set_ylabel("Magnitude")
    #splot.set_xlabel("Angles")
    #show()    
    
    #return array(magnitude_sums)
def similar_region_hog(interest_array, im_file, b, magnitude, stride=100):
    im_file_array = array(im_file)
    
    winning_region_array = []
    winning_distance = sys.maxsize
    
    start_time = timer()  
    
    for x in range(0, im_file.width, stride):
        for y in range(0, im_file.height, stride):
            region_array = im_file_array[x:x+len(interest_array),y:y+len(interest_array[0])]
            
            if region_array.shape != interest_array.shape:
                continue
            
            color_distance = 0            
            
            for i in range(3):        
                color_hog = hog(interest_array[:,:,i])
                dist = sqrt((color_hog ** 2 - interest_array_hog ** 2).sum())
                color_distance = max(color_distance, dist)
                
            if color_distance < winning_distance:
                winning_distance = color_distance
                winning_region_array = region_array            
                
    end_time = timer()    
    print('[HOG] Duration: ' + str(end_time - start_time))
    return winning_region_array               
    
def similar_region_pixel(interest_array, im_file, stride=100):
    im_file_array = array(im_file)

    winning_region_array = []
    winning_distance = sys.maxsize
    
    start_time = timer()  
    for x in range(0, im_file.width, stride):
        for y in range(0, im_file.height, stride):
            region_array = im_file_array[x:x+len(interest_array),y:y+len(interest_array[0])]
            
            if region_array.shape != interest_array.shape:
                continue
            
            color_distance = 0            
            
            for i in range(3):        
                dist = sqrt((interest_array[:,:,i] - region_array[:,:,i]).sum())
                color_distance = max(color_distance, dist)
                
            if color_distance < winning_distance:
                winning_distance = color_distance
                winning_region_array = region_array
                
    end_time = timer()    
    print('[Pixel] Duration: ' + str(end_time - start_time))
    return winning_region_array
    

im_file = Image.open("quijote1.jpg")
imshow(im_file) 
#region_corners = array(ginput(2))
#top_left = region_corners[0]
#bottom_right = region_corners[1]
# 488, 756
top_left = array([2064, 918])
bottom_right = array([2552, 1674])

#difference_hog("quijote1.jpg", "quijote2.jpg", top_left, bottom_right)
filename1 = "quijote1.jpg"
filename2 = "quijote1.jpg"

# Open the file and crop it
#im_file = Image.open(filename1)
#region_of_interest = crop_image(im_file, top_left, bottom_right)
im_file_compare = Image.open(filename2)
arr = array(im_file_compare)

#integral_images[] = get_hog_data(arr[:,:,i])

#interest_array = array(region_of_interest)
#interest_array_hog = hog(interest_array[:,:,2])
#distance_hog = sys.maxsize
#for i in range(2):    
#    color_hog = hog(interest_array[:,:,i])
#    dist = sqrt((color_hog ** 2 + interest_array_hog ** 2).sum())
    
#    if dist < distance_hog:
#        distance_hog = dist
 #       interest_array_hog = color_hog
#figure(3)
#imshow(similar_region_pixel(interest_array, im_file_compare))

#figure(4)
#imshow(similar_region_hog(interest_array, im_file_compare, b, magnitude))       
#file_1_hog = hog_grayscale(filename1, top_left, bottom_right, 12)
#file_2_hog = hog_grayscale(filename2, top_left, bottom_right, 12)
    
#diff = sqrt(square(file_1_hog - file_2_hog).sum())
show()