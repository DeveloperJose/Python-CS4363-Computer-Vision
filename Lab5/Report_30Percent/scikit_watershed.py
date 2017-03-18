# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 02:42:39 2016

@author: xeroj
"""
from PIL import Image
import cv2
import numpy as np
import pylab as plt
from skimage.filters import sobel
from skimage import morphology

im = np.array(Image.open("Hist-Level-01.jpg"))
im_g = np.array(Image.open("Hist-Level-01.jpg").convert("L"))

#elevation_map = sobel(im_g)
#
#ret, thresh = cv2.threshold(im_g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
## noise removal
#kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
## sure background area
#sure_bg = cv2.dilate(opening,kernel,iterations=3)
# 
## Finding sure foreground area
#dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#
## Finding unknown region
#sure_fg = np.uint8(sure_fg)
#unknown = cv2.subtract(sure_bg,sure_fg)
#
#ret, markers = cv2.connectedComponents(im_g)
#markers = markers + 1
#markers[unknown==255] = 0
#
#segmentation = morphology.watershed(elevation_map, markers)

plt.title("WIP")
plt.axis("off")
plt.imshow(label_im)