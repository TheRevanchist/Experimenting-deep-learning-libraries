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

def weight_variable(shape):
  """ This function initializes the weights of a neural network, by using 
  small random numbers drawn from a gaussian distribution """
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)
  
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

# create two placeholders for the data and the labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# reshape the image into a 4d array
x_image = tf.reshape(x, [-1,28,28,1]) 

# initialize the weights and the biases for the first conv layer                        
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])    

# perform the first convolutional layer followed by a max pool layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)    

# initialize the weights and the biases for the second conv layer  
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

# perform the second convolutional layer followed by a max pool layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) 

# initialize the weights and the biases for the fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# perform the fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  

# parameters for the dropout of the fully connected layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

# initialize the weights and the biases for the output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# perform the output layer
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  

# define the cost function (cross-entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
# like before, we train the network with the adam optimizer
train_step = tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)
# define the variables which measure the accuracy and then run the session
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

# for n iterations, do the training
for i in range(2000):
  batch = mnist.train.next_batch(128)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))         