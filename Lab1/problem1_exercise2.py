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
# Page 42-43, exercise 2
# Implement an unsharp masking operation by blurring an image then subtracting
# the blurred version from the original
# Try on both color and grayscale images
def unsharp_masking_grayscale(filename, gaussian_standard_deviation=15):
    im = array(Image.open(filename).convert('L'))
    # Begin calculation
    start_time = timer()

    im_blur = filters.gaussian_filter(im, gaussian_standard_deviation)
    im_unsharp_mask = im - im_blur

    # End calculation
    end_time = timer()
    print('[Unsharp Masking Grayscale] Duration: ' + str(end_time - start_time))

    return im_unsharp_mask

def unsharp_masking_color(filename, gaussian_standard_deviation=15):
    im_color = array(Image.open(filename))
    im_color_blur = zeros(im_color.shape)

    # Begin calculation
    start_time = timer()

    for i in range(3):
        im_color_blur[:,:,i] = filters.gaussian_filter(im_color[:,:,i], gaussian_standard_deviation)

    im_color_blur = uint8(im_color_blur)
    im_color_unsharp_mask = im_color - im_color_blur

    # End calculation
    end_time = timer()
    print('[Unsharp Masking Color] Duration: ' + str(end_time - start_time))

    return im_color_unsharp_mask

def problem1_exercise2(filename):
    grayscale_image = Image.open(filename).convert('L')
    color_image = Image.open(filename)

    grayscale_unsharp_mask = unsharp_masking_grayscale(filename)
    color_unsharp_mask = unsharp_masking_color(filename)

    # Grayscale figures
    figure(1)
    imshow(grayscale_image)

    figure(2)
    imshow(grayscale_unsharp_mask)

    # Color figures
    figure(3)
    imshow(color_image)

    figure(4)
    imshow(color_unsharp_mask)

    show()