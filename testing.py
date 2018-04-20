#This is the beginning of the VDT written by Vivian Bakiris 

import sys
import os
import skimage as ski
from skimage import io, color, util 
import imghdr
from matplotlib import pyplot as plt
import time 
import prox_tv as ptv
from skimage import img_as_uint
import warnings
from PIL import Image
import argparse



here = os.path.dirname(os.path.abspath(__file__))
filepath = here + "/" + 'photo.png'
filetype = imghdr.what(filepath)
X = io.imread(filepath)


X = ski.img_as_float(X)

print("Showing")

plt.imshow(X)
io.show()

io.imsave("outputImage.png", X)

print ("Opening Original")

image = Image.open('photo.png')
image.show()

print('Opening After')
image = Image.open('outputImage.png')
image.show()



