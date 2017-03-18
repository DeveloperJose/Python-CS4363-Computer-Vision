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

# Problem 3
# Rudin-Osher-Fatemi de-noising algorithm
def problem3(filename):
    import rof
    im = array(Image.open(filename).convert('L'))

    # Begin calculation
    start_time = timer()

    U, T = rof.denoise(im, im)

    # End calculation
    end_time = timer()
    print('[ROF De-Noising] Duration: ' + str(end_time - start_time))

    # Show figures
    figure()
    gray()
    imshow(U)
    show()