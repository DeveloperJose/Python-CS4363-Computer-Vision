# -*- coding: utf-8 -*-
"""
Course: CS 4365/5354 [Computer Vision]
Author: Chelsey J (with some minor modifications by Jose Perez)
Assignment: Lab 4
Instructor: Olac Fuentes
"""
import os
import cPickle
import numpy as np

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_cifar(batch_name="data_batch_1", path="."):
    """
    Loads CIFAR-10 files to 3d numpy arrays
    """
    file = os.path.join(path, batch_name)
    batches = os.path.join(path, 'batches.meta')

    data = unpickle(file).get('data')
    filename = np.array(unpickle(file).get('filenames'))
    labels = np.array(unpickle(file).get('labels'))
    label_names = np.array(unpickle(batches).get('label_names'))

    im_num = data.shape[0]
    row, col = 32, 32
    images = np.zeros((im_num, row, col, 3))

    for im in range(im_num):
        r_channel = data[im][0:1024].reshape((32, 32))
        g_channel = data[im][1024:2048].reshape((32, 32))
        b_channel = data[im][2048:3072].reshape((32, 32))
        images[im] = np.dstack([r_channel, g_channel, b_channel])

    return images, labels, label_names, filename


