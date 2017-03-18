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

# Problem 2
# Histogram Equalization Algorithm
def problem2(filename):
    import imtools

    im = array(Image.open(filename).convert('L'))
    # Begin calculation
    start_time = timer()

    im2, cdf = imtools.histeq(im)

    # End calculation
    end_time = timer()
    print('[Histogram Equalization] Duration: ' + str(end_time - start_time))

    imshow(im2)
    show()