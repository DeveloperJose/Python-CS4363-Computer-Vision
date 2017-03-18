# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:28:50 2016

@author: xeroj
"""

# ----------========== Evaluates the accuracy of neural network estimators
def evaluate_neural(dataset_flat, predict_lbls, true_lbls, label_names=None):
    (overall_total, z) = dataset_flat.shape
    print "Checking the accuracy of ",ALGORITHM," using {0} images".format(overall_total)
    
    overall_accurate_2 = predict_lbls[(true_lbls - predict_lbls) == 0].shape[0]
    # Keep track of how many we get right
    overall_accurate = 0
     
    for index in range(len(np.unique(predict_lbls))):
         # Get the current cluster
         cluster = dataset_flat[predict_lbls == index]
         cluster_total = cluster.shape[0]
     
         # Get the labels for the current cluster
         labels = true_lbls[predict_lbls == index]
     
         # Check if the cluster is empty
         if cluster_total == 0:
             if PRINT_CLUSTER_INFO:
                 print "Cluster {0} is empty.".format(index)
             continue
         
         # Select the majority as the cluster label
         majority = np.argmax(np.bincount(labels.flatten()))
         
         # How many did we cluster correctly?
         cluster_total_accurate = labels[labels == majority].shape[0]
 
         # Keep track of our overall accuracy
         overall_accurate += cluster_total_accurate
    
    print overall_accurate_2, overall_accurate
    return DECIMALS.format(float(overall_accurate_2) / overall_total * 100)