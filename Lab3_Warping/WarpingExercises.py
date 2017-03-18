# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Jose Perez [ID: 80473954]
Assignment: Warping Exercises
Instructor: Olac Fuentes
"""

from timeit import default_timer as timer
from PIL import Image
import scipy.ndimage
import numpy as np
from pylab import *

def H_from_points(fp,tp): 
    """ Find homography H, such that fp is mapped to tp using the linear DLT 
    method. Points are conditioned automatically. """
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
        A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0, tp[0][i]*fp[0][i],tp[0][i]
                    *fp[1][i],tp[0][i]] 
        A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1, tp[1][i]*fp[0][i],tp[1][i]
                    *fp[1][i],tp[1][i]]
    U,S,V = linalg.svd(A) 
    H = V[8].reshape((3,3))
    # decondition 
    H = dot(linalg.inv(C2),dot(H,C1))
    # normalize and return 
    return H / H[2,2]

def Haffine_from_points(fp,tp):
    """ Find H, affine transformation, such that 
        tp is affine transf of fp. """
    
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
        
    # condition points
    # --from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1]) 
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = dot(C1,fp)
    
    # --to points--
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = dot(C2,tp)
    
    # conditioned points have mean zero, so translation is zero
    A = concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
    U,S,V = linalg.svd(A.T)
    
    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]
    
    tmp2 = concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1) 
    H = vstack((tmp2,[0,0,1]))
    
    # decondition
    H = dot(linalg.inv(C2),dot(H,C1))
    
    return H / H[2,2]


################################################ Code Start
gray()
im_file = Image.open("wall.jpg").convert("L")
figure(1)
imshow(im_file)

im = array(im_file, dtype=np.float32)
fp = np.asarray(ginput(n=4))
width = 400
height = 600

(p0x, p0y) = fp[0]
(p1x, p1y) = fp[1]
(p2x, p2y) = fp[2]
(p3x, p3y) = fp[3]

fp = np.array([
    [p0x, p1x, p2x, p3x],
    [p0y, p1y, p2y, p3y],
    [1, 1, 1, 1]])    
    
(p0x, p0y) = (0,0) # Top-Left
(p1x, p1y) = (0, width) # Top-Right
(p2x, p2y) = (height, width) # Bottom-Right
(p3x, p3y) = (height, 0) # Bottom-Left

tp = np.array([
    [p0x, p1x, p2x, p3x],
    [p0y, p1y, p2y, p3y],
    [1, 1, 1, 1]]) 
    
H = H_from_points(fp, tp)
p_prime = dot(H, fp).astype(np.uint32)

canvas = zeros(im.shape)   

show()