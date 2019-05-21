import argparse
import sys
import tempfile
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = None
NITER = 200

def deepnn(x):
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
   # tf.cast(x_image, tf.float32)

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.variable_scope('conv1'):
    W_conv1 = tf.get_variable("W", [5, 5, 1, 32],
          initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 784), dtype=tf.float32), dtype=tf.float32)
    b_conv1 = tf.get_variable("b", initializer=tf.constant(0.1, shape=[32], dtype=tf.float32), dtype=tf.float32)
    z = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
   # z += b_conv1
    h_conv1 = tf.nn.relu(z + b_conv1)
    xxxxx = tf.identity(h_conv1, name='new_node')
  # Pooling layer - downsamples by 2X.
  with tf.variable_scope('pool1'):
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                     strides=[1, 2, 2, 1], padding='SAME')

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.variable_scope('conv2'):
    W_conv2 = tf.get_variable("W", [5, 5, 32, 64],
          initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / 32), dtype=tf.float32), dtype=tf.float32)
    b_conv2 = tf.get_variable("b", initializer=tf.constant(0.1, shape=[64], dtype=tf.float32), dtype=tf.float32)
    z = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
   # z += b_conv2
    h_conv2 = tf.nn.relu(z + b_conv2)

  # Pooling layer with 2nd convolutional layer
  with tf.variable_scope('pool2'):
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                     strides=[1, 2, 2, 1], padding='SAME')

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.variable_scope('fc1'):
    input_size = 7 * 7 * 64
    W_fc1 = tf.get_variable("W", [input_size, 1024],
          initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/input_size), dtype=tf.float32), dtype=tf.float32)
    b_fc1 = tf.get_variable("b", initializer=tf.constant(0.1, shape=[1024], dtype=tf.float32), dtype=tf.float32)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - regulate the complexity of the model
  with tf.variable_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.variable_scope('fc2'):
    W_fc2 = tf.get_variable("W", [1024, 10],
          initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/1024), dtype=tf.float32), dtype=tf.float32)
    b_fc2 = tf.get_variable("b", initializer=tf.constant(0.1, shape=[10], dtype=tf.float32), dtype=tf.float32)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)

  # Create the model
  
  x = tf.placeholder(tf.float32, [None, 784])

  # placeholder for true label
  #y_ = tf.placeholder(tf.int64, [None])
  y_ = tf.placeholder(tf.int32, [None])
  y_conv = tf.placeholder(tf.int32, [None])
  y_conv, keep_prob = deepnn(x)
 # tf.cast(y_conv, tf.int32)
  tf.cast(keep_prob, tf.int32)
  with tf.name_scope('loss'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1, output_type=tf.int32), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(NITER):

      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      save_path = saver.save(sess, "./Result_01_trained_net/model.ckpt")
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    #print(tf.global_variables(scope=None))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='../mnist_input_data/',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
