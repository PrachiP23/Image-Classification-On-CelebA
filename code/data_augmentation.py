
# coding: utf-8
# prachi patil
import numpy as np
import tensorflow as tf
from scipy import misc
import os
from scipy import ndimage
import matplotlib.pyplot as plt


whiteSpaceRegex = "\\s";
file = open("..\list_attr_celeba.txt", "r") 
file.readline()
file.readline()
path = "..\img_align_celeba\\";

arr = os.listdir(path);
celeb_images = np.zeros((len(arr), 784))
celeb_labels = []
for idx, filenames in enumerate(arr):
        image_file = path+filenames
        image = misc.imread(image_file)
        img_rot = misc.imrotate(image,30,interp='bilinear')
        plt.subplot(2, 2, 1)
        plt.imshow(img_rot, cmap=plt.cm.gray)

        imlr = np.flipud(image)
        plt.subplot(2, 2, 2)
        plt.imshow(imlr, cmap=plt.cm.gray)

        imf = np.fliplr(imlr)
        plt.subplot(2, 2, 3)
        plt.imshow(imf, cmap=plt.cm.gray)
        
        blur = ndimage.gaussian_filter(image, sigma=3)
        plt.subplot(2, 2, 4)
        plt.imshow(blur, cmap=plt.cm.gray)
        plt.show()





