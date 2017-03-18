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
# Page 42-43, exercise 1
# Take an image and apply Gaussian blur
# Plot the image contours for increasing values of standard deviation
def problem1_exercise1(filename):
    # Start calculation
    start_time = timer()

    im = array(Image.open(filename).convert('L'))
    im_blur_5 = filters.gaussian_filter(im, 5)
    im_blur_10 = filters.gaussian_filter(im, 10)
    im_blur_15 = filters.gaussian_filter(im, 15)
    im_blur_20 = filters.gaussian_filter(im, 20)

    # End calculation
    end_time = timer()
    print('[Problem 1 - Exercise 1] Duration: ' + str(end_time - start_time))

    # Show figures
    gray()
    figure(1)
    contour(im, origin='image', colors='purple')

    figure(2)
    contour(im_blur_5, origin='image', colors='purple')

    figure(3)
    contour(im_blur_10, origin='image', colors='purple')

    figure(4)
    contour(im_blur_15, origin='image', colors='purple')

    figure(5)
    contour(im_blur_20, origin='image', colors='purple')

    show()