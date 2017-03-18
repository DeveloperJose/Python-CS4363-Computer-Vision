# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Quiz - Feature Sets
Instructor: Olac Fuentes
"""
from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
import numpy as np
from pylab import *
from sklearn.cluster import *
import pdb

############################################# Problem 1
############################################# Color Histogram
def color_hist(I, n):
    hist = np.zeros((3, n+1))
    i = 0
    for color in range(0, 255, 255 / n):
        hist[0, i] = I[np.logical_and(I[:,:,0] >= color, I[:,:,0] < color + (255 / n))].sum()
        hist[1, i] = I[np.logical_and(I[:,:,1] >= color, I[:,:,1] < color + (255 / n))].sum()
        hist[2, i] = I[np.logical_and(I[:,:,2] >= color, I[:,:,2] < color + (255 / n))].sum()
        i += 1    

    return hist

#   Creates a bar graph for a given color histogram
def get_bar_graph(hist, n):
    fig = figure()
    splot = fig.add_subplot(111)
    
    splot.bar(range(len(hist)), hist, width=1)
    
    xTickMarks = [str(i) + "-"+ str(i + (255 / n)) for i in range(0, 255, 255 / n)]
    splot.set_xticklabels(xTickMarks)
    splot.set_ylabel("Magnitude")
    splot.set_xlabel("Color Values")
    
    return fig
############################################# Problem 2
############################################# 3D Color Histogram
def color_hist_3d(I, n):
    hist = np.zeros((n+1, n+1, n+1))
    step = 255 / n
    for x in range(I.shape[0]):
        for y in range(I.shape[1]):
                red = I[x,y, 0]
                green = I[x,y,1]
                blue = I[x, y, 2]
                hist[red / step, green / step, blue / step] += 1

    return hist

############################################# Problem 3
############################################# Grayscale Census Transform
def census_transform(I):
    result = np.zeros(I.shape)
    powers = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    for x in range(1, I.shape[0] - 1):
        for y in range(1, I.shape[1] - 1):
            region = I[x-1:x+2, y-1:y+2]
            mean = region.sum() / 9
            region[region <= mean] = 0
            region[region > mean] = 1

            result[x, y] = (region.flatten() * powers).sum()
            
    return result

############################################# Problem 4
############################################# Hamming Distance Sum
def hamming_sum(I1, I2):
    census_1 = census_transform(I1)
    census_2 = census_transform(I2)
    
    result = np.zeros(I1.shape)
    
    for x in range(1, I1.shape[0]-1):
        for y in range(1, I1.shape[1]-1):
            result[x, y] = hamming(int(census_1[x, y]), int(census_2[x, y]))
            
    return result

def hamming(x, y):
    if x == y:
        return 0
        
    if x % 2 != y % 2:
        return 1 + hamming(x/2, y/2)
        
    return hamming(x/2, y/2)
   
############################################# Test Program
I = np.array(Image.open("flower.jpg"), dtype=np.uint8)
I_gray = np.array(Image.open("flower.jpg").convert("L"), dtype=np.uint8)

# Problem 1 - Color histogram
n = 10
hist = color_hist(I, n)
graph_red = get_bar_graph(hist[0], n)
graph_green = get_bar_graph(hist[1], n)
graph_blue = get_bar_graph(hist[2], n)

# ============================== For testing the regular color histogram
#COLORS = np.array(["Red", "Green", "Blue"])
#BAR_COLORS = np.array(["r", "g", "b"])

#import pylab as plt
#def color_hist_graph(hist, n):
    #global COLORS
    #global BAR_COLORS
    
    #fig = plt.figure(figsize=(5,15))
    #for color in range(3):
        # 3 rows, 1 column
        #splot = fig.add_subplot(3, 1, color+1)
    
        #splot.bar(range(len(hist[color])), hist[color], width=1, color=BAR_COLORS[color])
        
        #xTickMarks = [str(i) + "-"+ str(i + (255 / n)) for i in range(0, 255, 255 / n)]
        #splot.set_xticklabels(xTickMarks)
        #splot.set_ylabel("Magnitude")
        #splot.set_xlabel("Color Values for " + COLORS[color])

    #return fig

#from PIL import Image    
#im = np.array(Image.open("flower.jpg"))
    
#hist = color_hist(temp, 10)
#fig = color_hist_graph(hist, 10)

#Iq = np.uint8(temp * 12.0 / 256.0)

#plt.show()

# Problem 2 - 3D color histogram
hist_3d = color_hist_3d(I, n)

# Problem 3 - Census
census = census_transform(I_gray)

# Problem 4 - Hamming
I1 = np.array(Image.open("left.png").convert("L"), dtype=np.uint8)
I2 = np.array(Image.open("right.png").convert("L"), dtype=np.uint8)

# Test Cases
print hamming(0b1111, 0b1111), "equals", 0
print hamming(0b1111, 0b1110), "equals", 1
print hamming(0b1111, 0b1001), "equals", 2
print hamming(0b1111, 0b0100), "equals", 3
print hamming(0b1111, 0b0000), "equals", 4
print hamming(0b11111111, 0b00110100), "equals", 5
print hamming(0b11111111, 0b00011000), "equals", 6
print hamming(0b11111111, 0b00001000), "equals", 7
print hamming(0b11111111, 0b00000000), "equals", 8
print hamming(0b111111111, 0b000000000), "equals", 9

sums = hamming_sum(I1, I2)

show()