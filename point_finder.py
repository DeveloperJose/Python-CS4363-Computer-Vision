# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 04:46:26 2016

@author: jgpd
"""
from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
import numpy as np
from pylab import *

im_file = Image.open("obj1.jpg")
imshow(im_file)
corners = np.asarray(ginput(n=4), dtype=np.float32)

print corners

show()