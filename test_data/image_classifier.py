#!/usr/bin/python3


import skimage
from skimage import io, transform
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image



# User modifiable input parameters
IMAGE                   = '5.png'
ROOT_PATH               = '/home/ubuntu/tfTutorial/'
IMAGE_PATH              = ROOT_PATH + 'test_data/' + IMAGE
IMAGE_DIM               = ( 28, 28 )


# Read & resize image [Image size is defined during training]
#imgOriginal = skimage.io.imread( IMAGE_PATH )

#img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True )
#imgResized = skimage.transform.resize( imgOriginal, IMAGE_DIM)
imgOriginal = Image.open(IMAGE_PATH)

#img = imgResized[ :, :, 2]

img = imgOriginal.resize(IMAGE_DIM, Image.ANTIALIAS)
img = img.convert('1')


fig = plt.figure()
a=fig.add_subplot(1,2,1)
plt.imshow(imgOriginal)

a=fig.add_subplot(1,2,2)
plt.imshow(img)

plt.show()



