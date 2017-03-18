# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 2
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image
from pylab import *
import numpy as np
from os import sys

from find_pixel_match import *

# ========================= Constants =========================

# Image to look for the object of interest
#interest_filename = "quijote1.jpg"
interest_filename = "obj1.jpg"

# Filenames of images to compare to
#compare_filenames = np.array(["quijote2.jpg", "quijote3.jpg", "quijote4.jpg"])
compare_filenames = np.array(["obj2.jpg", "obj3.jpg", "obj4.jpg"])

# Scaling factor
scale_factors = np.array([0.8, 0.6, 0.4, 1.2, 1.4, 1.6, 1.8])

# ========================= Object of Interest =========================
im_file = Image.open(interest_filename)
im = array(im_file)

# ========================= Region Selection =========================
# To let the user select the points you must use a normal Python console
#corners = np.asarray(ginput(n=4), dtype=np.int64)
#top_left = corners[0]
#bototm_right = corners[1]

# Glue coordinates in obj1.jpg
top_left = np.array([693, 63])
bottom_right = np.array([1101, 724])

# Quijote Coordinates in quijote1.jpg
#top_left = np.array([2064, 918])
#bottom_right = np.array([2552, 1674])

# ========================= Region Variables =========================
region_x = top_left[0]
region_y = top_left[1]
region_width = bottom_right[1] - top_left[1]
region_height = bottom_right[0] - top_left[0]
region = im[region_y:region_y+region_width, region_x:region_x+region_height]

# ========================= Calculations (Images) =========================

# Go through the comparison images
for i in range(len(compare_filenames)):
    # Get a filename
    filename = compare_filenames[i]

    print "-----===== [Pixel] Beginning of comparison using: ", filename

    image_file = Image.open(filename)
    # Create an image from the file
    image = np.array(image_file)
    
    # ----=== Compare using Pixels ===---
    print "Comparing using Pixels..."
    start_time = timer()
    
    (distance, best_match) = find_pixel_match(image, region)
    best_scale = 1.0
    
    end_time = timer()
    print "Matching took ", end_time - start_time
    print "Distance:  ", distance
    print "Best Match: ", best_match
    # If we aren't satisfied with our match then try to match
    # again but with the image scaled up or down    
    # -= Begin scaling =-
    for scale in scale_factors:
        print "---== Starting scale calculations, scale: ", scale
        # Calculate the new width
        new_width = int(image_file.size[0] * scale)
            
        # Keep aspect ratio for the new height
        wpercent = (new_width / float(image_file.size[0]))
        new_height = int((float(image_file.size[1]) * float(wpercent)))
        
        # Create new image
        scaled_image = np.array(image_file.resize((new_width, new_height), Image.ANTIALIAS))
        
        # Find the best match in the current scale
        print "Comparing with the scaled image using Pixels"
        start_time = timer()
        
        (curr_distance, curr_best_match) = find_pixel_match(image, region)
        
        end_time = timer()
        print "Matching took ", end_time - start_time
        print "Distance:  ", curr_distance
        print "Best Match: ", curr_best_match

        # Check if we found a better match
        if curr_distance < distance:
            distance = curr_distance
            best_match = curr_best_match
            best_scale = scale

    print "--== Finished matching"
    print "Best match (overall), ", best_match
    print "Best distance (overall), ", distance
    print "Best scale (overall), ", best_scale
    # --== Display results ==--
    fig = figure(2 + i)
    imshow(image)
    splot = fig.add_subplot(111)
    import matplotlib.patches as patches
    disp_x = (best_match[1]/best_scale)
    disp_y = (best_match[0]/best_scale)
    rect = patches.Rectangle((disp_x, disp_y),
                             region_height,
                             region_width,
                             linewidth=1,edgecolor='r',facecolor='none')
    splot.add_patch(rect)
    show()