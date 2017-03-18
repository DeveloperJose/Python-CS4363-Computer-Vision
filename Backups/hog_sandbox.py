from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
import numpy as np
from pylab import *
   
def H_from_points(fp,tp): 
    """ Find homography H, such that fp is mapped to tp using the linear DLT method. Points are conditioned automatically. """
    if fp.shape != tp.shape: 
        raise RuntimeError("number of points do not match")
        # condition points (important for numerical reasons) 
    # --from points-
    m = mean(fp[:2], axis =1) 
    maxstd = max(std(fp[:2], axis =1)) + 1e-9 
    C1 = diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd 
    C1[1][2] = -m[1]/maxstd 
    fp = dot(C1,fp)
    # --to points-
    m = mean(tp[:2], axis =1) 
    maxstd = max(std(tp[:2], axis =1)) + 1e-9
    
    C2 = diag([1/maxstd, 1/maxstd, 1]) 
    C2[0][2] = -m[0]/maxstd 
    C2[1][2] = -m[1]/maxstd 
    tp = dot(C2,tp)
    # create matrix for linear method, 2 rows for each correspondence pair 
    nbr_correspondences = fp.shape[1] 
    A = zeros((2*nbr_correspondences,9)) 
    for i in range(nbr_correspondences): 
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0, tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]] 
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1, tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]
    U,S,V = linalg.svd(A) 
    H = V[8].reshape((3,3))
    # decondition 
    H = dot(linalg.inv(C2),dot(H,C1))
    # normalize and return 
    return H / H[2,2]

#p = np.array([[0,0],[0, im_file.width],[im_file.height, im_file.width],[im_file.height, 0]])
#p = np.array([[0,0],[im_file.height, 0],[im_file.height, im_file.width],[0, im_file.width]])
#p = np.array([[0, im_file.height, im_file.height, 0],[0, 0, im_file.width, im_file.width]])
#p = vstack((p, np.ones(p.shape[1])))
#corners = corners.reshape((corners.shape[1], corners.shape[0]))
#corners = vstack((corners, np.ones(corners.shape[1])))

#H = H_from_points(p, corners)
#H = p * np.linalg.pinv(p)

#result = p * H
#imshow(result)

def get_hog_data(im_array, number_of_bars=12):
    from scipy.ndimage.filters import convolve
    convolution_matrix =  array([[-1, 0, 1], [-1,0,1], [-1, 0, 1]])   
    
    imx = zeros(im_array.shape)
    imx = convolve(im_array, convolution_matrix)
      
    imy = zeros(im_array.shape)
    imy = convolve(im_array, transpose(convolution_matrix))   
       
    magnitude = sqrt(imx**2+imy**2)
    #direction = degrees(2 * arctan2(1, 1/imx)-imy)
    direction = arctan2(imy, imx) * 180 / pi

    # Put angles together depending on the number of bars  
    angle_distance = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_distance)
    bins = []
    for angle in angle_range:
        #current_bin = logical_and(direction>=angle, direction<angle+30)    
        current_bin = logical_and(direction>=angle, direction<angle+30)        
        bins.append(current_bin)

    integral_images = []    
    for i in range(len(bins)):
        current_bin = bins[i]
        integral = magnitude
        integral[not current_bin] = 0
        integral.cumsum(axis=0).cumsum(axis=1)
        integral_images.append(integral)
    
    return integral_images
    
def hog(filename, top_left, bottom_left, number_of_bars):
    # Calculate the rectangle to crop the image    
    distance = bottom_left - top_left
    #crop_rect = append(top_left, distance)
    
    # Open the file and crop it
    im_file = Image.open(filename).convert('L')
    im_file_array = array(im_file)
    #im = array(im_file.crop(crop_rect).convert('L'))
    x = top_left[0]
    y = top_left[1]
    im = im_file_array[x:x+distance[0],y:y+distance[1]]
            
    
    start_time = timer()      
    
    # Convolution
    # Calculate magnitude and direction
    # Direction will be in degrees
    from scipy.ndimage.filters import convolve
    #convolution_matrix =  array([[-1, 0, 1], [-1,0,1], [-1, 0, 1]])   
    convolution_matrix =  array([[-1, 0, 1]])  
    
    imx = zeros(im.shape)
    imx = convolve(im, convolution_matrix)
      
    imy = zeros(im.shape)
    imy = convolve(im, transpose(convolution_matrix))   
       
    magnitude = sqrt(imx**2+imy**2)
    direction = arctan2(imy, imx) * 180 / pi
  
    # Put angles together depending on the number of bars  
    angle_distance = 360 / number_of_bars
    angle_range = xrange(0, 360, angle_distance)
    bins = []
    for angle in angle_range:
        current_bin = logical_and(direction>=angle, direction<angle+30)    
        bins.append(current_bin)
    
    # Sum the magnitudes of the angle bins    
    magnitude_sums = []
    
    for i in range(len(bins)):    
        magnitude_for_bin = magnitude[bins[i]]
        total_magnitude = magnitude_for_bin.sum()
        magnitude_sums.append(total_magnitude)        
    
    return magnitude_sums    
    
    end_time = timer()    
    print('[Normal HOG] Duration: ' + str(end_time - start_time)) 
    
    # Prepare the bar graph
    fig = figure(2)
    splot = fig.add_subplot(111)
    
    splot.bar(range(len(magnitude_sums)), magnitude_sums, width=1)
    
    xTickMarks = [str(i) + "-"+ str(i + angle_distance) for i in angle_range]
    splot.set_xticklabels(xTickMarks)
    splot.set_ylabel("Magnitude")
    splot.set_xlabel("Angles")
    show()
    
    # Normal hog("flower.jpg", array([0, 0]), array([100, 100]), 12)
    # hog("quijote1.jpg", array([0, 0]), array([2500, 2500]), 12)
#im_file = Image.open("quijote1.jpg").convert("L")
#top_left = array([2064, 918])
#bottom_right = array([2552, 1674])

#im_array = array(im_file)
#integrals = get_hog_data(im_array)

#normal = hog("quijote1.jpg", top_left, bottom_right, 12)

#x = top_left[0]
#y = top_left[1]
#distance = bottom_right - top_left
#region = array(im_file)[x:x+distance[0], y:y+distance[1]]
#bar1_fast = integrals[0][x:x+distance[0], y:y+distance[1]]

#width = distance[0]
#height = distance[1]
#row = x
#column = y

#integral = integrals[0]
#A = integral[width:row,height:column]
#B = integral[width:row,:column-height]
#C = integral[:row-width,height:column]
#D = integral[:row-width,:column-height]

#Sum = A + D - B - C