# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Lab 2
Instructor: Olac Fuentes
"""
import numpy as np
from os import sys
# ========================= Constants =========================
# What amount of difference between HOGs is enough to stop?
difference_threshold = 500

# Stride of our comparison search
stride = 1

def find_hog_match(image_hog, region_hog):
    distance = sys.maxsize
    best_match = (-1, -1)
    
    for x in range(1, image_hog.shape[2]-1, stride):
        for y in range(1, image_hog.shape[1]-1, stride):
            comp_hog = image_hog[:, y-1, x-1]
            
            if (region_hog.shape != comp_hog.shape):
                continue
            
            #diff = ((region_hog - comp_hog) ** 2).sum()
            diff = np.linalg.norm(region_hog  - comp_hog)        
            
            if diff < distance:
                distance = diff
                best_match = (y, x)
                
            if distance <= difference_threshold:
                return (distance, best_match, True)
    
    return (distance, best_match, False)