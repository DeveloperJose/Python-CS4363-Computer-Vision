# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:21:32 2016

@author: Olac Fuentes
"""

import numpy as np
from PIL import Image
from pylab import *
#from indexImage import *
from scipy.ndimage import filters

im_l = np.array(Image.open('left.png'))
im_r = np.array(Image.open('right.png'))

figure(0)
imshow(uint8(im_l))
figure(1)
imshow(uint8(im_r))

max_disp = 200
min_disp = 30
# width for ncc
wid = 20

# array to hold depth planes 
m,n,c = im_l.shape

ts = zeros((m,n))
disp_map= zeros((m,n))
im_diff = zeros((m,n,2))+1e10
for displ in range(min_disp,max_disp):
    
    if displ % 5 == 0:
        print "Batch of 5, step: ", displ
        
    s = zeros((m,n))
    #for i in range(3):
    #    temp = roll(im_l[:,:,i],-displ)-im_r[:,:,i]
    #    temp = temp*temp
#       filters.gaussian_filter(temp,wid,0,ts) 
    #    filters.uniform_filter(temp,wid,ts)
    #    s = s+ts
     
    temp = roll(im_l,-displ)-im_r
    temp = temp*temp
    
    filters.uniform_filter(temp,wid,ts[:,:,0])
    s += ts

    filters.uniform_filter(temp,wid,ts[:,:,1])
    s += ts  
    
    filters.uniform_filter(temp,wid,ts[:,:,2])
    s += ts   
    
    im_diff[:,:,1] = s
    ind = im_diff[:,:,1] < im_diff[:,:,0] 
    disp_map[ind] = displ
    im_diff[:,:,0] = np.amin(im_diff, axis=2)
        
figure(2)
imshow(uint8(disp_map))

