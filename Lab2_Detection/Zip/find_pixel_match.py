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
# Stride of our comparison search
stride = 1

def find_pixel_match(image, region):
    distance = sys.maxsize
    best_match = (-1, -1)
    
    for x in range(1, image.shape[2]-1, stride):
        for y in range(1, image.shape[1]-1, stride):
            comp = image[x:x+region.shape[0], y:y+region.shape[1], :]
            
            if (region.shape != comp.shape):
                continue
            
            diff = ((comp - region) ** 2).sum()      
            
            if diff < distance:
                distance = diff
                best_match = (y, x)
    
    return (distance, best_match)