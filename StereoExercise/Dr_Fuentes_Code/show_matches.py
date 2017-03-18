# -*- coding: utf-8 -*-
"""
Stereo Matching Program 

@author: Olac Fuentes
"""

import matplotlib.pyplot as plt
from pylab import ginput


def round(x):
    return int(x+.5)

def my_ginput():
# Returns row and column, instead of x,y
    p_l = np.asarray(ginput(1))
    p_l = np.array([p_l[0,1],p_l[0,0]])
    return p_l

w =40

plt.figure(0)
plt.imshow(im_l)
plt.axis('image')
p_l =my_ginput()


rows = [-w, w, w, -w , -w ]+ p_l[0]
cols = [-w, -w, w, w, -w] +p_l[1]
plt.plot(cols,rows,"-r")

disp = disp_map[p_l[0],p_l[1]]

p_r = np.array([p_l[0],p_l[1]-disp])

rows = [-w, w, w, -w , -w ]+ p_r[0]
cols = [-w, -w, w, w, -w] +p_r[1]

figure(1)
imshow(im_r)
plt.plot(cols,rows,"-r")
plt.axis('image')   