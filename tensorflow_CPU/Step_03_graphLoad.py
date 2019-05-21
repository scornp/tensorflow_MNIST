import argparse
import sys
import tempfile
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

IMAGE = '3xxB.png'

# User modifiable input parameters
ROOT_PATH               = './'
META_PATH               = ROOT_PATH + 'Result_02_IOAdded/IOAdded.meta'
CHK_PATH                = ROOT_PATH + 'Result_02_IOAdded/'
IMAGE_PATH              = ROOT_PATH + '../test_data/' + IMAGE
LABELS_FILE_PATH        = ROOT_PATH + '../test_data/labels.txt'
IMAGE_DIM               = ( 28, 28 )

imgOriginal = Image.open(IMAGE_PATH)
img = imgOriginal
img = img.resize(IMAGE_DIM)
print(img)
im2arr = np.array(img).reshape([784, 1])
im2arr = im2arr.astype( np.float32 )

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(META_PATH)
saver.restore(sess,tf.train.latest_checkpoint(CHK_PATH))

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()

x = graph.get_tensor_by_name("input:0")

feed_dict ={x:im2arr}

op_to_restore = graph.get_tensor_by_name("output:0")
print(sess.run(op_to_restore,feed_dict))
out = sess.run(op_to_restore,feed_dict)
print(out)

fig = plt.figure()
a=fig.add_subplot(1,3,1)

x = range( 0, 10 )
y = out[0]
width = 0.35

plt.bar(x, y, width)
plt.xlabel('digit')
plt.ylabel('prediction')

a=fig.add_subplot(1,3,2)
plt.imshow(imgOriginal)

a=fig.add_subplot(1,3,3)
plt.imshow(img, cmap=plt.get_cmap('gray_r'))
#plt.savefig("output.png")
plt.show()

"""
#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

print sess.run(op_to_restore,feed_dict)
#This will print 60 which is calculated 
#using new values of w1 and w2 and saved value of b1.
"""
