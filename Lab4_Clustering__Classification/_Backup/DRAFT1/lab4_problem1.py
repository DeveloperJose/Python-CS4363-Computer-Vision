# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 4
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image
import numpy as np
import pylab as plt
from math import ceil
from sklearn.cluster import KMeans

# =============== Variables
filename = "landscape_s.jpg"
k_min = 1
k_max = 12
k_step = 1

# =============== Constants
DECIMALS = '{:.3f}' # Set the decimals for the timer results
IMAGE_COLUMNS = 3.0 # (Floating pt) Number of images to show per row in results 

# ============================== Load image
im_original = np.array(Image.open(filename), dtype=np.float64)

# Flatten for k-means
(w, h, c) = im_original.shape
im_flat = np.reshape(im_original, (w * h, c))

# Prepare figure
figure = plt.figure(0, figsize=(4, 7))
figure.suptitle(filename + " | k-means" )
nrows = ceil(k_max / k_step / IMAGE_COLUMNS)

start_time = timer()
for k in range(k_min, k_max+k_step, k_step):
    temp = im_flat.copy()
    # Run K-Means
    start_time = timer()
    k_means = KMeans(n_clusters = k).fit(temp)
    #print "K-Means took ", time_format.format(timer() - start_time) , "s for k =", k
    
    start_time = timer()
    for cluster in range(k):
        temp[k_means.labels_ == cluster] = k_means.cluster_centers_[cluster]
    
    #print "Replacing took ", time_format.format(timer() - start_time), "s"

    subplot = figure.add_subplot(nrows+1, IMAGE_COLUMNS, k / k_step)
    subplot.set_xticks(())
    subplot.set_yticks(())
    subplot.set_xlabel("k="+str(k))
    subplot.imshow(np.reshape(temp, (w, h, c)).astype(np.uint8))
    
plt.show()

print "Took ", DECIMALS.format(timer() - start_time) , "s"