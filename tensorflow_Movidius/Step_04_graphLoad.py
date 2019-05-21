#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# How to classify images using DNNs on Intel Neural Compute Stick (NCS)

import mvnc.mvncapi as mvnc
import skimage
from skimage import io, transform
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

IMAGE = '3xxB.png'

# User modifiable input parameters
ROOT_PATH               = './'
GRAPH_PATH              = ROOT_PATH + 'Result_03_movidiusGraph/IOAdded.graph' 
IMAGE_PATH              = ROOT_PATH + '../test_data/' + IMAGE
LABELS_FILE_PATH        = ROOT_PATH + '../test_data/labels.txt'
IMAGE_DIM               = ( 28, 28 )

# ---- Step 1: Open the enumerated device and get a handle to it -------------

# Look for enumerated NCS device(s); quit program if none found.
devices = mvnc.enumerate_devices()
if len( devices ) == 0:
	print( 'No devices found' )
	quit()

# Get a handle to the first enumerated device and open it
mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)
device = mvnc.Device( devices[0] )
device.open()

# ---- Step 2: Load a graph file onto the NCS device -------------------------

# Read the graph file into a buffer
with open( GRAPH_PATH, mode='rb' ) as f:
	blob = f.read()

# Load the graph buffer into the NCS
graph = mvnc.Graph('graph')
fifoIn, fifoOut = graph.allocate_with_fifos(device, blob)

# ---- Step 3: Offload image onto the NCS to run inference -------------------

# Read & resize image [Image size is defined during training]
imgOriginal = skimage.io.imread( IMAGE_PATH )

#img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True )
imgResized = skimage.transform.resize( imgOriginal, IMAGE_DIM)

#img = imgResized[ :, :, 3].reshape(784, 1)

#print(img)

#type(imgOriginal)
#img = img.reshape(28, 28)
img = imgResized

img = img.astype( np.float32 )


# Load the image as a half-precision floating point array
graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, img, 'output')
#graph.LoadTensor( img.astype( np.float16 ), 'user object' )

# ---- Step 4: Read & print inference results from the NCS -------------------

# Get the results from NCS
output, userobj = fifoOut.read_elem()
#output, userobj = graph.GetResult()

print( output )

fig = plt.figure()
a=fig.add_subplot(1,3,1)

x = range( 0, 10 )
y = output
width = 0.35

plt.bar(x, y, width)
plt.xlabel('digit')
plt.ylabel('prediction')

a=fig.add_subplot(1,3,2)
plt.imshow(imgOriginal)

a=fig.add_subplot(1,3,3)
plt.imshow(imgResized, cmap=plt.get_cmap('gray_r'))
plt.show()

print('\n------- predictions --------')

labels = np.loadtxt( LABELS_FILE_PATH, str, delimiter = '\t' )

order = output.argsort()[::-1][:6]

for i in range( 0, 4 ):
	print ('prediction ' + str(i) + ' is ' + labels[order[i]])

# ---- Step 5: Unload the graph and close the device -------------------------


fifoIn.destroy()
fifoOut.destroy()
graph.destroy()
device.close()

# ==== End of file ===========================================================

