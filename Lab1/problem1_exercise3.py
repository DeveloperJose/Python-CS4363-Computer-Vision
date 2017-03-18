# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 1
Instructor: Olac Fuentes
Last Modification: September 2, 2016 by Jose Perez
"""
from timeit import default_timer as timer
from scipy.ndimage import filters
from PIL import Image
from numpy import *
from pylab import *
# Page 42-43, exercise 3
# Quotient image
# Divide the image with a blurred version
# I / (I * G_std)
def quotient_image(filename, gaussian_standard_deviation=15):
    im = array(Image.open(filename).convert('L'))
    # Begin calculation
    start_time = timer()

    im_blur = filters.gaussian_filter(im, gaussian_standard_deviation)
    im_quotient = im / (im * im_blur)

    # End calculation
    end_time = timer()
    print('[Quotient Image] Duration: ' + str(end_time - start_time))

    return im_quotient

def problem1_exercise3(filename):
    example = quotient_image(filename)
    imshow(example)
    show()
