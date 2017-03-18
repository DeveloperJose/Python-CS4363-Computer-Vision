# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:01:50 2016

@author: jgpd
"""
import numpy as np
from PIL import Image
from pylab import *

im_file = Image.open("quijote1.jpg")
scale = 1/.2

figure(1)
imshow(im_file)

# Upscale 3 times
scale_img = im_file.copy()
basewidth = int(im_file.size[0] * scale)
wpercent = (basewidth / float(im_file.size[0]))
hsize = int((float(im_file.size[1]) * float(wpercent)))
img = im_file.resize((basewidth, hsize), Image.ANTIALIAS)
#for i in range(1, 4):
 #   newSize = np.dot(i / scale, im_file.size).astype(np.uint32)
  #  scale_img_2 = scale_img.resize(newSize)
 
   # figure(i + 2)
    #imshow(scale_img_2)