# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Stereo Image Exercise
Instructor: Olac Fuentes
"""
from PIL import Image
from pylab import *
import numpy
from scipy.ndimage import filters
from timeit import default_timer as timer

def plane_sweep_ncc(im_l, im_r, start, steps, wid):
  '''Find disparity image using normalized cross-correlation.'''

  m, n, c = im_l.shape  # Must match im_r.shape.

  mean_l = numpy.zeros(im_l.shape)
  mean_r = numpy.zeros(im_l.shape)
  s = numpy.zeros(im_l.shape)
  s_l = numpy.zeros(im_l.shape)
  s_r = numpy.zeros(im_l.shape)

  dmaps = numpy.zeros((m, n, steps))

  filters.uniform_filter(im_l, wid, mean_l)
  filters.uniform_filter(im_r, wid, mean_r)

  norm_l = im_l - mean_l
  norm_r = im_r - mean_r

  for disp in range(steps):
    filters.uniform_filter(numpy.roll(norm_l, -disp - start) * norm_r, wid, s)
    filters.uniform_filter(numpy.roll(norm_l, -disp - start) *
                           numpy.roll(norm_l, -disp - start), wid, s_l)
    filters.uniform_filter(norm_r * norm_r, wid, s_r)

    dmaps[:, :, disp] = s / numpy.sqrt(s_l * s_r)

  return numpy.argmax(dmaps, axis=2)


def plane_sweep_ssd(im_l, im_r, start, steps, wid):
    '''Find disparity image using sum of squared differences'''
    
    m, n = im_l.shape  # Must match im_r.shape.
    
    s = numpy.zeros(im_l.shape)
    
    dmaps = numpy.zeros((m, n, 2))
    
    for disp in range(steps):
        #filters.gaussian_filter((numpy.roll(im_l, -disp - start) - im_r) ** 2, wid, 0, s)
        filters.uniform_filter((numpy.roll(im_l, -disp - start) - im_r) ** 2, wid, s)
        dmaps[:, :, disp] = s
    
    return numpy.argmin(dmaps, axis=2)

def plane_sweep_ssd_color_old(im_l, im_r, start, steps, wid):
    '''Find disparity image using sum of squared differences.'''
    '''Take the average of the colors every step'''
    m, n, c = im_l.shape  # Must match im_r.shape.

    s = numpy.zeros(im_l.shape)

    dmaps = numpy.zeros((m, n, steps))
    
    print "Beginning step calculation with total steps: ", steps
    for disp in range(steps):
        filters.uniform_filter((numpy.roll(im_l, -disp - start) - im_r) ** 2, wid, s)
        
        result = (s[:,:,0] + s[:,:,1] + s[:,:,2]) / 3                
            
        if disp % 10 == 0:
            print "Batch of 10 completed, current step: ", disp
            
        dmaps[:, :, disp] = result

    return numpy.argmin(dmaps, axis=2)
    
def plane_sweep_ssd_color(im_l, im_r, start, steps, wid):
    '''Find disparity image using sum of squared differences.'''
    '''Average of colors taken before'''
    im_l = (im_l[:,:,0] + im_l[:,:,1] + im_l[:,:,2]) / 3
    im_r = (im_r[:,:,0] + im_r[:,:,1] + im_r[:,:,2]) / 3
    m, n = im_l.shape  # Must match im_r.shape.

    s = numpy.zeros(im_l.shape)

    dmaps = numpy.zeros((m, n, steps))    
    
    print "Beginning step calculation with total steps: ", steps
    for disp in range(steps):
        filters.uniform_filter((numpy.roll(im_l, -disp - start) - im_r) ** 2, wid, s)
        
        dmaps[:, :, disp] = s
        
        if disp % 10 == 0:
            print "Batch of 10 completed, current step: ", disp
            
    return numpy.argmin(dmaps, axis=2)

im_l = numpy.array(Image.open("piano_left.png")) 
im_r = numpy.array(Image.open("piano_right.png"))

# starting displacement and steps 
steps = 50
start = 2
wid = 6
#for wid in range(6, 15, 3):  
start_time = timer() 
res = plane_sweep_gauss_color(im_l,im_r,start,steps,15)

end_time = timer()
print "SSD Duration: ", end_time - start_time

figure(1)
imshow(im_l)

#figure(2)
#imshow(im_r)

figure(3)
imshow(res)
