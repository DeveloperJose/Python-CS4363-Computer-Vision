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
from sklearn.cluster import *
from lib_cifar import load_cifar

def color_hist(I, n):
    hist = np.zeros((3, n+1))
    i = 0
    for color in range(0, 255, 255 / n):
        hist[0, i] = I[np.logical_and(I[:,:,0] >= color, I[:,:,0] < color + (255 / n))].sum()
        hist[1, i] = I[np.logical_and(I[:,:,1] >= color, I[:,:,1] < color + (255 / n))].sum()
        hist[2, i] = I[np.logical_and(I[:,:,2] >= color, I[:,:,2] < color + (255 / n))].sum()
        i += 1    

    return hist

# ----------========== Parameters
SHOW_CLUSTER_IMAGES = True
IMAGE_COLUMNS = 5

CLUSTERS = 10
HISTOGRAMS = 10
# ----------========== Constants
time_format = '{0:.3f}' # Set the decimals for the timer results


start_time = timer()
(training, training_lbls, training_filenames) = load_cifar(batch_name="data_batch_1", path="cifar")
print "Took ", time_format.format(timer() - start_time) , "s for CIFAR loading"



(overall_total, im_w, im_h, im_c) = training.shape

training_hists = np.zeros((overall_total, 3, HISTOGRAMS+1))
(hist_total, hist_colors, hist_bars) = training_hists.shape

i = 0
for image_index in range(overall_total):
    training_hists[i] = color_hist(training[image_index], HISTOGRAMS)
    i += 1
    
training_flat = np.reshape(training_hists, (hist_total, hist_colors * hist_bars))

print "Performing K-Means for {0} images.".format(overall_total) 

start_time = timer()
k_means = KMeans(n_clusters = CLUSTERS)
k_means.fit(training_flat)

print "K-Means took {0}s for k = {1}".format(time_format.format(timer() - start_time), CLUSTERS)

# Keep track of how many we get right
overall_accurate = 0

print "Finding accuracy of clusters"
for cluster_index in range(CLUSTERS):
    # Get the current cluster
    cluster = training_flat[k_means.labels_ == cluster_index]
    cluster_total = cluster.shape[0]

    # Get the real labels
    labels = training_lbls[k_means.labels_ == cluster_index]

    # Check if the cluster is empty
    if cluster_total == 0:
        print "Cluster {0} is empty.".format(cluster_index)
        continue
    
    # Get the majority as the cluster label
    majority = np.argmax(np.bincount(labels.flatten()))
    
    # How many did we cluster correctly?
    cluster_total_accurate = labels[labels == majority].shape[0]

    # Figure out the accuracy
    accuracy = time_format.format(cluster_total_accurate / float(cluster_total) * 100)

    # Keep track of our overall progress
    overall_accurate += cluster_total_accurate
    
    if SHOW_CLUSTER_IMAGES:
        figure = plt.figure(cluster_index)
        figure.suptitle("Cluster {0}, Total {1}, Accuracy {2}".format(cluster_index, cluster_total, accuracy))
        
        #subplot = figure.add_subplot(nrows+2, IMAGE_COLUMNS, 1)
        #subplot.imshow(np.reshape(k_means.cluster_centers_[cluster_index], (im_w, im_h)))
        
        for image_index in range(cluster_total):
            #cluster_total_half = int(cluster.shape[0] / 2)
            #subplot(nrows, ncols, plot_number)
            nrows = cluster_total / IMAGE_COLUMNS
            subplot = figure.add_subplot(nrows+2, IMAGE_COLUMNS, 
                                         image_index + 2)
            subplot.set_xticks(())
            subplot.set_yticks(())
            
            subplot.imshow(training[image_index])

        plt.show()
    else:
        print "Cluster {0}, Total {1}, Accuracy {2}".format(cluster_index, cluster_total, accuracy)

accuracy = time_format.format(float(overall_accurate) / overall_total * 100)
print "K-Means Overall Accuracy: {0}%".format(accuracy)