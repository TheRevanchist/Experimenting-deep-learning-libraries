# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:49:54 2016

@author: revan
"""

""" On this script, we are going to build a convolutional neural
network, and then train and test it on the MNIST dataset """

# get the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# import tensorflow and build the session
import tensorflow as tf
sess = tf.InteractiveSession()

# misc
import numpy as np
  
def bias_variable(shape):
  """ This function initializes the biases of a neural network to zero values """
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  """ This function performs the convolution pass """
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """ This function does a 2 by 2 max pooling """
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  


# initialize everything for the first layer
number_of_pixels = 784
width, height = int(np.sqrt(number_of_pixels)), int(np.sqrt(number_of_pixels))
number_of_channels = 1
number_of_classes = 10
number_of_filters_1 = 32
width_filter_1, height_filter_1, depth_filter_1 = 3, 3, number_of_channels                     
W_conv1 = tf.get_variable("W_conv1", shape=[width_filter_1, height_filter_1, depth_filter_1,
           number_of_filters_1],
           initializer=tf.contrib.layers.variance_scaling_initializer())                           
b_conv1 = bias_variable([number_of_filters_1])   

# create two placeholders for the data and the labels
x = tf.placeholder(tf.float32, shape=[None, number_of_pixels])
y_ = tf.placeholder(tf.float32, shape=[None, number_of_classes])
keep_prob = tf.placeholder(tf.float32)
# reshape the image into a 4d array
x_image = tf.reshape(x, [-1, width, height, number_of_channels]) 

# initialize everything for the second layer 
number_of_filters_2 = 64
width_filter_2, height_filter_2, depth_filter_2 = 3, 3, number_of_filters_1
W_conv2 = tf.get_variable("W_conv2", shape=[width_filter_2, height_filter_2, depth_filter_2,
           number_of_filters_2],
           initializer=tf.contrib.layers.variance_scaling_initializer())                            
b_conv2 = bias_variable([number_of_filters_2]) 

# initialize everything for the third layer
number_of_filters_3 = 128
width_filter_3, height_filter_3, depth_filter_3 = 3, 3, number_of_filters_2
W_conv3 = tf.get_variable("W_conv3", shape=[width_filter_3, height_filter_3, depth_filter_3,
           number_of_filters_3],
           initializer=tf.contrib.layers.variance_scaling_initializer())                             
b_conv3 = bias_variable([number_of_filters_3])                            

# initialize everything for the fourth layer (fc7 layer)
number_of_neurons_fc = 1024
# 7 = (28:2):2
final_width, final_height = 7, 7
W_fc1 = tf.get_variable("W_fc1", shape=[final_width * final_height * number_of_filters_3,
           number_of_neurons_fc],
           initializer=tf.contrib.layers.variance_scaling_initializer()) 
b_fc1 = bias_variable([number_of_neurons_fc])

# initialize the weights and the biases for the fifth (output) layer
W_fc2 = tf.get_variable("W_fc2", shape=[number_of_neurons_fc,number_of_classes],
           initializer=tf.contrib.layers.variance_scaling_initializer()) 
b_fc2 = bias_variable([number_of_classes])

# perform a batch-norm, followed by the first convolutional layer followed by a max pool laye
x_image_bn = tf.contrib.layers.batch_norm(x_image)
h_conv1 = tf.nn.relu(conv2d(x_image_bn, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)    

# perform dropout, batchnorm, the second convolutional layer followed by a max pool layer
h_pool1_drop = tf.nn.dropout(h_pool1, keep_prob) 
h_pool1_drop_bn = tf.contrib.layers.batch_norm(h_pool1_drop)
h_conv2 = tf.nn.relu(conv2d(h_pool1_drop_bn, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) 

# perform dropout, batchnorm, the third convolutional layer (careful: no max pooling here)
h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob) 
h_pool2_drop_bn = tf.contrib.layers.batch_norm(h_pool2_drop)
h_conv3 = tf.nn.relu(conv2d(h_pool2_drop_bn, W_conv3) + b_conv3)

# perform the fully connected layer (together with dropout and batchnorm)
h_conv3_drop = tf.nn.dropout(h_conv3, keep_prob) 
h_conv3_flat = tf.reshape(h_conv3_drop, [-1, final_height * final_height * number_of_filters_3])
h_conv3_flat_bn = tf.contrib.layers.batch_norm(h_conv3_flat)
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)  

# perform the output layer (together with dropout and batchnorm)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 
h_fc1_drop_bn = tf.contrib.layers.batch_norm(h_fc1_drop)
y_conv = tf.matmul(h_fc1_drop_bn, W_fc2) + b_fc2  

# define the cost function (cross-entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
# like before, we train the network with the adam optimizer
train_step = tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)
# define the variables which measure the accuracy and then run the session
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

# for n iterations, do the training
for i in range(4000):
  batch = mnist.train.next_batch(128)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))         