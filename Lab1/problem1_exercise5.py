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
from numpy import *

# Page 42-43, exercise 5
# Gradient direction and magnitude
# Detect lines in an image
# Estimate extent of the lines and parameters
# Plot lines overlaid on the image
def problem1_exercise5(filename, threshold=50):
    im = array(Image.open(filename).convert('L'))
    background = Image.open(filename).convert('RGBA')

    # Begin calculation
    start_time = timer()

    # Sobel
    imx_sobel = zeros(im.shape)
    filters.prewitt(im, 1, imx_sobel)

    imy_sobel = zeros(im.shape)
    filters.prewitt(im, 0, imy_sobel)

    magnitude = sqrt(imx_sobel**2 + imy_sobel**2)

    # Filter (AKA set a threshold)
    low_values = magnitude < threshold
    high_values = magnitude >= threshold
    magnitude[low_values] = 255 # White for values below threshold
    magnitude[high_values] = 0 # Black for values above threshold

    magnitude_color = Image.fromarray(magnitude).convert('RGBA')
    magnitude_color_arr = array(magnitude_color)
    red, green, blue, alpha = magnitude_color_arr.T

    black_areas = (red == 0) & (blue == 0) & (green == 0)
    white_areas = (red == 255) & (blue == 255) & (green == 255)
    magnitude_color_arr[:,:,:][black_areas.T] = (255, 0, 0, 255) # Convert black to red for outline
    magnitude_color_arr[:,:,:][white_areas.T] = (0, 0, 255, 0) # Make white transparent

    # Overlay over the original image in color
    overlay = Image.fromarray(magnitude_color_arr)
    background.paste(overlay, None, overlay)

    # End calculation
    end_time = timer()
    print('[Problem 1 - Exercise 5] Duration: ' + str(end_time - start_time))

    imshow(background)