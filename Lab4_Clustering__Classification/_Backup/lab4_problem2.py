# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 4
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
import numpy as np
import pylab as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from math import ceil

from lib_mnist import load_mnist
from lib_cifar import load_cifar
from lib_features import pixel_intensity
from lib_features import color_histogram
from lib_features import gradient_histogram
from lib_features import daisy_feature

# MNIST
# ----------
# Daisy Features
#   k = 10
#   Entire dataset (Train: 60000, Test: 10000)
#   (interrupted after 1752s)   
#   180s/3 min to compute feature
# ----------
# Pixel Intensities
#   k = 10
#   Entire dataset
#   147s total time
#   141.826s to train k-means
#   59.073% training accuracy
#   59.460% testing accuracy
#
#   k = 20
#   Entire dataset
#   192s total time
#   185.103s to train k-means
#   70.990% training accuracy
#   71.470% testing accuracy
# ----------
# Histograms of Gradients
#   k = 10
#   Entire dataset
#   48.378s total time
#   32.608s to compute features
#   9.713s to train k-means
#   29.868% training accuracy
#   30.370% testing accuracy
# 
#   k=20
#   Entire dataset
#   61.300s total time
#   29.287s to train k-means
#   37.898% training accuracy
#   39.510% testing accuracy
# ----------

# CIFAR10
# ----------
# Daisy Features
#   k = 10
#   1 batch (Train: 7500, Test: 2500)
#   316s/55 min total time
#   45s to compute feature
#   270s/4 min to train k-means
#   24.307% training accuracy
#   24.440% testing accuracy
# ----------
# Color Histograms
#   k = 10
#   5 batches (Train: 37500, Test: 12500)
#   57.851s total time
#   45.722s to compute features
#   8.995s to train k-means 
#   18.480% training accuracy
#   18.928% testing accuracy
# ----------
# Histogram of Gradients
#   k = 10
#   5 batches (Train: 37500, Test: 12500)
#   55.816s total time
#   47.322s to compute feature   
#   5.319s to train k-means
#   24.568% training accuracy
#   24.808% testing accuracy

# ----------========== Global Parameters
SHOW_CLUSTER_IMAGES = False # Show images or just print results?
SHOW_CENTROIDS_ONLY = False # IF showing images, show only the centroids? (Only for PIXEL)
IMAGE_COLUMNS = 25.0 # (Floating pt) Number of images to show per row in results 

CLUSTERS = 20 # Number of K clusters
CURRENT_DATASET = "MNIST" # {MNIST, CIFAR10}
CIFAR_BATCHES = 1 # Number of CIFAR-10 batches to use
FEATURE = "HOG" # {PIXEL, COLORHIST, HOG, DAISY}
ALGORITHM = "KMEANS" # {KMEANS, AFFINITYPROP, MINIBATCHKMEANS, MEANSHIFT}

HISTOGRAMS = 12 # Number of bars in the histogram features (if used)

CURRENT_FIGURE = 0 # Prevents overlapping figures
DECIMALS = '{:.3f}' # Set the decimals for the timer results

# ----------========== K-Means Evaluation
def evaluate_accuracy(cluster_centers, predict_lbls, dataset, dataset_flat, true_lbls, label_names=None):
    global SHOW_CLUSTER_IMAGES
    global IMAGE_COLUMNS
    global CLUSTERS
    global CURRENT_DATASET
    global ALGORITHM
    global CURRENT_FIGURE
    global DECIMALS
    
    if CURRENT_DATASET == "MNIST":
        (overall_total, w, h) = dataset.shape
    else:
        (overall_total, w, h, c) = dataset.shape
        
    print "Checking the accuracy of ",ALGORITHM," using {0} images".format(overall_total)
    # Keep track of how many we get right
    overall_accurate = 0
    
    for cluster_index in range(CLUSTERS):
        # Get the current cluster
        cluster = dataset_flat[predict_lbls == cluster_index]
        cluster_total = cluster.shape[0]
    
        # Get the labels for the current cluster
        labels = true_lbls[predict_lbls == cluster_index]
    
        # Check if the cluster is empty
        if cluster_total == 0:
            print "Cluster {0} is empty.".format(cluster_index)
            continue
        
        # Select the majority as the cluster label
        majority = np.argmax(np.bincount(labels.flatten()))
        
        # How many did we cluster correctly?
        cluster_total_accurate = labels[labels == majority].shape[0]

        # Keep track of our overall accuracy
        overall_accurate += cluster_total_accurate
    
        # Figure out the accuracy
        accuracy = DECIMALS.format(cluster_total_accurate / float(cluster_total) * 100)
        
        if CURRENT_DATASET == "MNIST":
            title = "Cluster #{0}, Digit: {1}, Total: {2}, Correct: {3}, Accuracy: {4}%".format(cluster_index, majority, cluster_total, cluster_total_accurate, accuracy)
        else:
            title = "Cluster #{0}, {1}, Total: {2}, Correct: {3}, Accuracy: {4}%".format(cluster_index, label_names[majority], cluster_total,  cluster_total_accurate, accuracy)
        
        if SHOW_CLUSTER_IMAGES:
            figure = plt.figure(CURRENT_FIGURE)
            figure.suptitle(title)
            
            CURRENT_FIGURE += 1
            offset = 1
            
            nrows = ceil(cluster_total / IMAGE_COLUMNS)
                
            if FEATURE == "PIXEL":
                if SHOW_CENTROIDS_ONLY:
                    subplot = figure.add_subplot(1, 1, 1)
                else: # Show centroid in the middle of the first row
                    subplot = figure.add_subplot(nrows+1, IMAGE_COLUMNS, ceil(IMAGE_COLUMNS / 2.0))
                
                subplot.set_xticks(())
                subplot.set_yticks(())
                if CURRENT_DATASET == "MNIST":
                    subplot.imshow(np.reshape(cluster_centers[cluster_index], (w, h)))
                else:
                    subplot.imshow(np.reshape(cluster_centers[cluster_index], (w, h, c)))
                
                offset = 6
                nrows += 1
            
            if not SHOW_CENTROIDS_ONLY:
                # Show all the images in the cluster depending on the number of columns selected
                for image_index in range(cluster_total):
                    subplot = figure.add_subplot(nrows+1, IMAGE_COLUMNS, image_index+offset)
                    
                    subplot.set_xticks(())
                    subplot.set_yticks(())
                    
                    if CURRENT_DATASET == "MNIST":
                        subplot.imshow(np.reshape(dataset[image_index], (w, h)))
                    else:
                        subplot.imshow(np.reshape(dataset[image_index], (w, h, c)))
                    
                plt.show()
        else:
            print title
    
    return DECIMALS.format(float(overall_accurate) / overall_total * 100)

# =========================================================================================
# =========================================================================================
# =========================================================================================
# =========================================================================================
# =========================================================================================
program_start = timer()
print "-----========== Dataset Loading"
print "Dataset:", CURRENT_DATASET, ", Feature:", FEATURE, ", Algorithm: ", ALGORITHM

if CURRENT_DATASET == "MNIST":
    if SHOW_CLUSTER_IMAGES:
        plt.gray() # MNIST images are grayscale
        
    start_time = timer()
    (training, training_lbls) = load_mnist(dataset="training", path="mnist")
    (testing, testing_lbls) = load_mnist(dataset="testing", path="mnist")
    dataset_label_names = None
    dataset_filenames = None
    
elif CURRENT_DATASET == "CIFAR10":
    print "Loading", CIFAR_BATCHES, "CIFAR-10 batches"
    start_time = timer()
    # Load the first batch
    (dataset, dataset_lbls, dataset_label_names, dataset_filenames) = load_cifar(batch_name="data_batch_1", path="cifar")
    
    # Append all the batches requested into one dataset
    for index in range(2, CIFAR_BATCHES+1):
        (dataset2, dataset_lbls2, dataset_label_names2, dataset_filenames2) = load_cifar(batch_name="data_batch_"+str(index), path="cifar")
    
        dataset = np.append(dataset, dataset2, axis=0)
        dataset_lbls = np.append(dataset_lbls, dataset_lbls2, axis=0)
        dataset_label_names = np.append(dataset_label_names, dataset_label_names2, axis=0)
        dataset_filenames = np.append(dataset_filenames, dataset_filenames2, axis=0)
        
    # Split into training and testing sets
    split = int(len(dataset)*0.75)
    training = dataset[:split]
    training_lbls = dataset_lbls[:split]
    
    testing = dataset[split:]
    testing_lbls = dataset_lbls[split:]
        
else:
    raise Exception("CURRENT_DATASET is invalid. Only [MNIST, CIFAR10] are valid datasets.", CURRENT_DATASET)

print "Took ", DECIMALS.format(timer() - start_time) , "s to load and split", CURRENT_DATASET
print "Training:", training.shape[0], ",Testing:", testing.shape[0]
# =========================================================================================
print "-----========== Computing Features:", FEATURE
start_time = timer()
if FEATURE == "PIXEL":
    training_flat = pixel_intensity(training)
    testing_flat = pixel_intensity(testing)
elif FEATURE == "COLORHIST":
    training_flat = color_histogram(training, HISTOGRAMS)
    testing_flat = color_histogram(testing, HISTOGRAMS)
elif FEATURE == "HOG":
    training_flat = gradient_histogram(training, HISTOGRAMS)
    testing_flat = gradient_histogram(testing, HISTOGRAMS)
elif FEATURE == "DAISY":
    training_flat = daisy_feature(training)
    testing_flat = daisy_feature(testing)        
else:
    raise Exception("FEATURE is invalid. Only [PIXEL, COLORHIST, HOG, DAISY] are valid features.", FEATURE)

print "Took ", DECIMALS.format(timer() - start_time) , "s to compute features"
# =========================================================================================
print "-----========== Clustering Training:", ALGORITHM
print "Performing",ALGORITHM,"training for {0} images.".format(training.shape[0]) 
start_time = timer()
if ALGORITHM == "AFFINITYPROP":
    k_means = AffinityPropagation(n_clusters=CLUSTERS, n_jobs=1)
elif ALGORITHM == "MINIBATCHKMEANS":
    k_means = MiniBatchKMeans(n_clusters=CLUSTERS, n_jobs=1)
elif ALGORITHM == "MEANSHIFT":
    k_means= MeanShift(n_clusters=CLUSTERS, n_jobs=1)
else:
    k_means = KMeans(n_clusters=CLUSTERS, n_jobs=1)
    
k_means.fit(training_flat)
print ALGORITHM, "took ",DECIMALS.format(timer() - start_time),"s for k = ", CLUSTERS

# =========================================================================================
print "-----==========",ALGORITHM,"Training Accuracy Evaluation"
start_time = timer()
training_accuracy = evaluate_accuracy(k_means.cluster_centers_, k_means.labels_, training, training_flat, training_lbls, dataset_label_names)
print ALGORITHM, "Training Accuracy:", training_accuracy, "%"
print "Took ",DECIMALS.format(timer() - start_time),"s for k = ", CLUSTERS

# =========================================================================================
print "-----========== ",ALGORITHM,"Testing"
print "Performing K-Means testing for {0} images.".format(testing.shape[0]) 
start_time = timer()
testing_prediction_lbls = k_means.predict(testing_flat)
print "Took ",DECIMALS.format(timer() - start_time),"s for k = ", CLUSTERS

# =========================================================================================
print "-----==========",ALGORITHM,"Testing Accuracy Evaluation"
start_time = timer()
testing_accuracy = evaluate_accuracy(k_means.cluster_centers_, testing_prediction_lbls, testing, testing_flat, testing_lbls, dataset_label_names)
print ALGORITHM, "Testing Accuracy:", testing_accuracy, "%"
print "Took ",DECIMALS.format(timer() - start_time),"s for k = ", CLUSTERS

print "Program ran for ",DECIMALS.format(timer() - program_start),"s"