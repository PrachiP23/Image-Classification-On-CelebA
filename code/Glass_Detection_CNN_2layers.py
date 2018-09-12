
# coding: utf-8

# prachi patil

import numpy as np
import tensorflow as tf
from scipy import misc
import os
from skimage.color import rgb2gray
import argparse
import sys
import tempfile
import matplotlib.pyplot as plt 
import tensorflow as tf
import time


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
        img = image.resize(28,28)
        img = np.asarray(img, dtype=np.float32)
        celeb_images[idx,0:784] = img.flatten()
        string = file.readline()
        splitStr = string.split()
        
        if int(splitStr[16]) == 1:
            labels = [0,1]
        else:
            labels = [1,0]
        celeb_labels.append(labels)

celeb_images = np.asarray(celeb_images, dtype=np.float32)
celeb_labels = np.asarray(celeb_labels, dtype=np.float32)



#shuffle the data
randomize = np.arange(len(celeb_images))
np.random.shuffle(randomize)
celeb_images_random = celeb_images[randomize]
celeb_labels_random = celeb_labels[randomize]

#divide the data into 2 sets: training and testing
train_images = celeb_images_random[:int(len(celeb_images_random)*0.9)]
train_labels = celeb_labels_random[:int(len(celeb_labels_random)*0.9)]

test_images = celeb_images_random[int(len(celeb_images_random)*0.9) :]
test_labels = celeb_labels_random[int(len(celeb_labels_random)*0.9) :]

print('deepnn')

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 2 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# Create the model
x = tf.placeholder(tf.float32, [None, 784])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 2])

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
#tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
#tf.summary.scalar('accuracy', accuracy)

#merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter('../graphs/train1',
 #                                     sess.graph)
#test_writer = tf.summary.FileWriter('../graphs/test1')
    
graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())



sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
batch_size = 100
training_epochs = 10
total_batch = int(train_images.shape[0] / batch_size)
print(total_batch)

for epoch in range(training_epochs):
    for i in range(total_batch):
        batch_x, batch_y = train_images[i * batch_size: (i + 1) * batch_size], train_labels[i * batch_size: (i + 1) * batch_size]
        if i % 100 == 0: 
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
print('test accuracy %g' % sess.run(accuracy, feed_dict={x:test_images, y_: test_labels, keep_prob: 1.0}))





