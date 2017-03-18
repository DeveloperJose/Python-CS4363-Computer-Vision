# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 05:14:50 2016

@author: xeroj
"""
from PIL import Image
from pylab import *
import numpy as np
from os import sys
# --== Display results ==--
image = Image.open("quijote2.jpg")
best_match = (1407, 1721)
region_height = 488
region_width = 756
fig = figure(2)
imshow(image)
splot = fig.add_subplot(111)
import matplotlib.patches as patches
rect = patches.Rectangle(((best_match[0]/1.0)+region_width, # x
                          (best_match[1]/1.0)-region_height), # y
                         region_height,
                         region_width,
                         linewidth=1,edgecolor='r',facecolor='none')
splot.add_patch(rect)
show()

# quijote2
# 2731, 4983
# scale 1.6