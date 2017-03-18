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
from math import ceil
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Requires scikit-learn 0.18 or above
from sklearn.neural_network import MLPClassifier

# Deep neural networks
#   Can be installed with
#       pip install scikit-neuralnetwork
#   Requires Theano
#       http://deeplearning.net/software/theano/install_windows.html#install-windows
from sknn.mlp import Classifier as MLPClassifier2
from sknn.mlp import Layer
from sknn.mlp import Convolution
from sknn.mlp import Native

# https://github.com/Lasagne/Lasagne
# Deep neural networks
import lasagne.layers as lasagne_l

from lib_mnist import load_mnist
from lib_cifar import load_cifar
from lib_features import pixel_intensity
from lib_features import color_histogram
from lib_features import gradient_histogram
from lib_features import daisy_feature

import warnings
warnings.filterwarnings("ignore")

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
# ----------========== Global Parameters
# Show all the images in each cluster?
# Only for clustering algorithms
SHOW_CLUSTER_IMAGES = False

# If showing images, show only the centroids?
SHOW_CENTROIDS_ONLY = False 

# (Floating pt) Number of images to show per row in results 
IMAGE_COLUMNS = 25.0 

# Print the accuracy of each cluster? 
# Not recommended for AffinityProp and MeanShift   
PRINT_CLUSTER_INFO = False 

# Number of K clusters (Only for KMeans and MiniBatchKMeans)
CLUSTERS = 10 
# {MNIST, CIFAR10}
CURRENT_DATASET = "MNIST" 

# Number of CIFAR-10 batches to use (Max: 5)
CIFAR_BATCHES = 5

# Number of bars in the histogram features (if used)
HISTOGRAMS = 16 

# {PIXEL, COLORHIST, HOG, DAISY}
FEATURE = "PIXEL"

# Clustering: {KMEANS, AFFINITYPROP, MINIBATCHKMEANS, MEANSHIFT}
# Neural Networks: {MLP, MLP2, CNN}
# SVM: {LINEARSVC, SVC9}
ALGORITHM = "CNN"

# Required for SVMs
FEATURE_SCALING = True

# Number of units in neural networks
NEURONS = 100

# Epochs to train CNN
EPOCHS = 10

# Should we deskew the images if possible?
PERFORM_DESKEW = False

# Set the decimals for the timer results
DECIMALS = '{:.3f}'

# Uses small datasets for debugging
DEBUGGING = False

# To prevent overlapping figures
CURRENT_FIGURE = 0
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
# ----------========== Evaluates the accuracy of clustering classifiers
# ----------========== Needed for majority labeling of k-means
def evaluate_clustering(dataset, dataset_flat, predict_lbls, true_lbls, cluster_centers, label_names=None):
    global SHOW_CLUSTER_IMAGES
    global SHOW_CENTROIDS_ONLY
    global IMAGE_COLUMNS
    global PRINT_CLUSTER_INFO
    
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
            if PRINT_CLUSTER_INFO:
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
            title = "Cluster #{0}, Digit: {1}, {2}/{3} correct, Accuracy: {4}%".format(cluster_index, majority, cluster_total_accurate, cluster_total, accuracy)
        else:
            title = "Cluster #{0}, {1}, {2}/{3} correct, Accuracy: {4}%".format(cluster_index, label_names[majority], cluster_total_accurate,  cluster_total, accuracy)
        
        # Can only show centers for clustering algorithms
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
        elif PRINT_CLUSTER_INFO:
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
    
    if DEBUGGING: # 60,000 max
        training = training[0:500]
        training_lbls = training_lbls[0:500]

        testing = testing[0:500]
        testing_lbls = testing_lbls[0:500]
    
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
    
    if DEBUGGING:
        training = dataset[0:200]
        training_lbls = dataset_lbls[0:200]
    
        testing = dataset[200:400]
        testing_lbls = dataset_lbls[200:400]
    else:    
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
    training_flat = pixel_intensity(training, PERFORM_DESKEW)
    testing_flat = pixel_intensity(testing, PERFORM_DESKEW)
    
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

if FEATURE_SCALING:
    scaler = StandardScaler()
    scaler.fit(training_flat)
    training_flat = scaler.transform(training_flat)
    
    scaler.fit(testing_flat)
    testing_flat = scaler.transform(testing_flat)        

print "Took ", DECIMALS.format(timer() - start_time) , "s to compute features"
# =========================================================================================
print "-----========== Training:", ALGORITHM
print "Performing",ALGORITHM,"training for {0} images.".format(training.shape[0]) 
start_time = timer()

if ALGORITHM == "AFFINITYPROP":
    k_means = AffinityPropagation(affinity="euclidean")
    k_means.fit(training_flat)
    CLUSTERS = len(k_means.cluster_centers_)
    
elif ALGORITHM == "MINIBATCHKMEANS":
    k_means = MiniBatchKMeans(n_clusters=CLUSTERS)
    k_means.fit(training_flat)
    
elif ALGORITHM == "MEANSHIFT":
    k_means= MeanShift()
    k_means.fit(training_flat)
    CLUSTERS = len(k_means.cluster_centers_)
    
elif ALGORITHM == "MLP":
    k_means = MLPClassifier(hidden_layer_sizes=(NEURONS, ), 
                            activation='relu', solver='adam', 
                            alpha=0.0001, batch_size='auto', 
                            learning_rate='constant', learning_rate_init=0.001, 
                            power_t=0.5, max_iter=10, shuffle=True, 
                            random_state=None, tol=0.0001, verbose=False, 
                            warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                            early_stopping=False, validation_fraction=0.1, 
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    k_means.fit(training_flat, training_lbls.flatten())
    
    print "Neurons: ", NEURONS
    
elif ALGORITHM == "MLP2":
    k_means = MLPClassifier2(
        layers=[
            Layer("Rectifier", units=NEURONS),
            Layer("Softmax")],
        learning_rate = 0.1,
        learning_rule='nesterov',
        learning_momentum=0.9,
        batch_size=300,
        valid_size=0.0,
        f_stable=0.001,
        n_stable=10,
        n_iter=10)
    k_means.fit(training_flat, training_lbls.flatten())
    
    print "Neurons: ", NEURONS

elif ALGORITHM == "CNN":
    k_means = MLPClassifier2(
        layers=[
            #Native(lasagne_l.Conv2DLayer(incoming=(32,3,3), num_filters=32, filter_size=(3, 3)), nonlinearity=lasagne_nl.rectify),
            Convolution('Rectifier',channels=32,kernel_shape=(3,3),border_mode="same"),
            Native(lasagne_l.DropoutLayer,p=0.2),
            #Native(lasagne_l.Conv2DLayer,num_filters=32, filter_size=(3, 3)),
            Convolution('Rectifier',channels=32,kernel_shape=(3,3),border_mode="same"),
            Native(lasagne_l.MaxPool2DLayer,pool_size=(2,2)),
            Native(lasagne_l.FlattenLayer),
            Native(lasagne_l.DenseLayer,num_units=NEURONS),
            Native(lasagne_l.DropoutLayer,p=0.5),
            
            #Convolution('Rectifier', channels=24, kernel_shape=(3, 3), dropout=0.05),   
            #Layer('Rectifier', units=NEURONS, dropout=0.05),
            Layer('Softmax'),
            ],
        learning_rate=0.01,
        learning_rule='sgd',
        learning_momentum=0.9,
        batch_size=32,
        weight_decay=0.01/EPOCHS,
        valid_size=0.0,
        f_stable=0.001,
        n_stable=10,
        n_iter=EPOCHS)    
    
    k_means.fit(training_flat, training_lbls.flatten())
    
    print "Neurons: ", NEURONS
    
elif ALGORITHM == "LINEARSVC":
    k_means = LinearSVC()
    k_means.fit(training_flat, training_lbls.flatten())

elif ALGORITHM == "SVC9":
    k_means = SVC(degree=9)
    k_means.fit(training_flat, training_lbls.flatten())
    
else:
    k_means = KMeans(n_clusters=CLUSTERS, n_jobs=1)
    k_means.fit(training_flat)
    

print ALGORITHM, "took ",DECIMALS.format(timer() - start_time), "s"

# =========================================================================================
print "-----========== ",ALGORITHM,"Prediction"
print "Performing ",ALGORITHM," prediction for {0} images.".format(testing.shape[0]) 
start_time = timer()

predict_lbls = k_means.predict(testing_flat)
    
print "Took ",DECIMALS.format(timer() - start_time),"s"

# =========================================================================================
print "-----==========",ALGORITHM,"Testing Accuracy Evaluation"
start_time = timer()

if ALGORITHM == "MLP" or ALGORITHM == "LINEARSVC" or ALGORITHM == "CNN" or ALGORITHM == "SVC9" or ALGORITHM == "MLP2" or ALGORITHM == "AUTOENCODER":
    testing_accuracy = k_means.score(testing_flat, testing_lbls) * 100
else:
    testing_accuracy = evaluate_clustering(testing, testing_flat, predict_lbls, testing_lbls, k_means.cluster_centers_, dataset_label_names)

print ALGORITHM, "Testing Accuracy:", testing_accuracy, "%"
print "Took ",DECIMALS.format(timer() - start_time),"s"

print "Program ran for ",DECIMALS.format(timer() - program_start), "s"