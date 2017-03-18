# -*- coding: utf-8 -*-
from timeit import default_timer as timer
from PIL import Image
from scipy.ndimage import filters
from scipy import cluster
import numpy as np
from pylab import *

from matplotlib.colors import LinearSegmentedColormap

im_file = Image.open("wall.jpg").convert("L")
im = np.array(im_file, dtype=np.uint8)

cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }
        
blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)


figure(1)
imshow(np.uint8(blue_red1(im) * 255))