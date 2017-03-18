from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
import numpy as np
from pylab import *
from scipy.ndimage.filters import convolve
from scipy import signal

im_array = None
imx = None
imy = None
magnitude = None
direction = None
start_time = None
angle_distance = None
angle_range = None
fast_bins = None
angle = None
current_bin = None
fast_integral_images = None
fast_magnitude_sums = None

im_file = Image.open("quijote1.jpg").convert("L")
number_of_bars = 12

#top_left = array([1000, 1000])
#bottom_right = array([1500, 1500])

top_left = array([2064, 893])
bottom_right = array([2608, 1706])

#top_left = array([1535,926])
#bottom_right = array([2007,1625])

x = top_left[0]
y = top_left[1]
width = bottom_right[1] - top_left[1]
height = bottom_right[0] - top_left[0]
#im_array = array([[1,2,1,2,2,1],[1,3,0,2,1,3],[1,1,1,2,2,1],[0,1,1,0,2,2],[1,2,2,1,0,1]])
#region_array = array([[2,2],[0,2]])
#im_array = array(im_file)[y:y+width, x:x+height]
region_array = array(im_file)[y:y+width, x:x+height]

convolution_matrix =  array([[-1, 0, 1]])   

region_imx = zeros(region_array.shape)
region_imx = convolve(region_array, convolution_matrix)
  
region_imy = zeros(region_array.shape)
region_imy = convolve(region_array, transpose(convolution_matrix))   
   
region_magnitude = sqrt(region_imx**2+region_imy**2)
region_direction = arctan2(region_imy, region_imx) * 180 / pi

# Normal HOG
start_time = timer()
region_angle_distance = 360 / number_of_bars
region_angle_range = xrange(0, 360, region_angle_distance)
region_bins = []
for angle in region_angle_range:
    current_bin = logical_and(region_direction>=angle, region_direction<angle+30)    
    region_bins.append(current_bin)

# Sum the magnitudes of the angle bins    
magnitude_sums = []

for i in range(len(region_bins)):    
    region_temp = region_magnitude.copy()
    region_temp[invert(region_bins[i])] = 0
    total_magnitude = region_temp.sum()

    magnitude_sums.append(total_magnitude)          

end_time = timer()    
print('[Normal HOG] Duration: ' + str(end_time - start_time)) 

# Prepare the bar graph
#==============================================================================
# fig = figure(2)
# splot = fig.add_subplot(111)
# 
# splot.bar(range(len(magnitude_sums)), magnitude_sums, width=1)
# 
# xTickMarks = [str(i) + "-"+ str(i + angle_distance) for i in angle_range]
# splot.set_xticklabels(xTickMarks)
# splot.set_ylabel("Magnitude")
# splot.set_xlabel("Angles")
# show()
#==============================================================================

#magnitude_sums = None
################################################### Integral HOG
################################################### Region
# Put angles together depending on the number of bars  
#==============================================================================
# start_time = timer()
# 
# angle_distance = 360 / number_of_bars
# angle_range = xrange(0, 360, angle_distance)
# bins = []
# for angle in angle_range:
#     #current_bin = logical_and(direction>=angle, direction<angle+30)    
#     current_bin = logical_or(direction<angle, direction>=angle+30)        
#     bins.append(current_bin)
# 
# s_integral_images = []    
# fast_magnitude_sums = []
# for i in range(len(bins)):
#     current_bin = bins[i]
#     integral = magnitude.copy()
#     integral[current_bin] = 0
#     fast_magnitude_sums.append(integral.sum())
#     integral.cumsum(axis=0).cumsum(axis=1)
#     s_integral_images.append(integral)
# 
# # integral[width-1, height-1]
# 
# end_time = timer()    
# print('[Integral HOG] Duration: ' + str(end_time - start_time)) 
#==============================================================================

# Prepare the bar graph
#==============================================================================
# fig = figure(3)
# splot = fig.add_subplot(111)
# 
# splot.bar(range(len(fast_magnitude_sums)), fast_magnitude_sums, width=1)
# 
# xTickMarks = [str(i) + "-"+ str(i + angle_distance) for i in angle_range]
# splot.set_xticklabels(xTickMarks)
# splot.set_ylabel("Magnitude")
# splot.set_xlabel("Angles")
# show()
#==============================================================================

################################################### Integral HOG
################################################### Whole Image
################################################### Grayscale
im_array = None
imx = None
imy = None
magnitude = None
direction = None
start_time = None
angle_distance = None
angle_range = None
fast_bins = None
angle = None
current_bin = None
fast_integral_images = None
fast_magnitude_sums = None

im_array = array(im_file)  

imx = zeros(im_array.shape)
imx = convolve(im_array, convolution_matrix)
  
imy = zeros(im_array.shape)
imy = convolve(im_array, transpose(convolution_matrix))   
   
magnitude = sqrt(imx**2+imy**2)
direction = arctan2(imy, imx) * 180 / pi

start_time = timer()

angle_distance = 360 / number_of_bars
angle_range = xrange(0, 360, angle_distance)
fast_bins = []
    
for angle in angle_range:
    current_bin = logical_and(direction>=angle, direction<angle+30)    
    #angle_current_bin = logical_or(direction<angle, direction>=angle+30)        
    fast_bins.append(current_bin)

fast_integral_images = []    
fast_magnitude_sums = []
for i in range(len(fast_bins)):
    fast_current_bin = invert(fast_bins[i])
    integral = magnitude.copy()
    integral[fast_current_bin] = 0
    #integral = np.pad(integral, (1, 1), 'constant', constant_values=(0))
    #print "Wow", i, integral.sum()    
    #integral = integral.cumsum(axis=0).cumsum(axis=1)
    fast_integral_images.append(integral)

(row, column) = im_array.shape
(height, width) = region_array.shape
integral = fast_integral_images[0]
A = integral[width:,height:]
B = integral[width:row,:column-height]
C = integral[:row-width,height:column]
D = integral[:row-width,:column-height]
Sum = A + D - B - C

a = integral[y,x]
b = integral[y,x-2]
c = integral[y-2,x]
d = integral[y-2,x-2]
print a - b - c + d

print "Sum: ", Sum[bottom_right[1], bottom_right[0]]
print "magnitude_sums[0]: ", magnitude_sums[0]
print "?: ", (Sum[bottom_right[1], bottom_right[0]] + Sum[top_left[1], top_left[0]])/2
print "fix: ", integral[y:y+height, x:x+width].sum()

end_time = timer()    
print('[Integral HOG Whole] Duration: ' + str(end_time - start_time)) 

figure(2)
img1 = region_magnitude.copy()
img1[invert(region_bins[0])] = 0
img1 = region_direction
img1 = signal.convolve2d(region_array, convolution_matrix)
#img1 = region_array
imshow(img1)
#print "img1: ", img1.sum()

figure(3)
img2 = integral[y:y+height, x:x+width]
img2 = direction[y:y+height, x:x+width]
img2 = signal.convolve2d(im_array, convolution_matrix)
#img2 = im_array[y:y+height, x:x+width]
imshow(img2)

print array_equal(region_array, im_array[y:y+height, x:x+width])
#print "img2: ", img2.sum()
# Prepare the bar graph
#==============================================================================
# fig = figure(4)
# splot = fig.add_subplot(111)
# 
# splot.bar(range(len(fast_magnitude_sums)), fast_magnitude_sums, width=1)
# 
# xTickMarks = [str(i) + "-"+ str(i + angle_distance) for i in angle_range]
# splot.set_xticklabels(xTickMarks)
# splot.set_ylabel("Magnitude")
# splot.set_xlabel("Angles")
# show()
#==============================================================================