# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:06:18 2016

@author: revan
"""

# get the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# import tensorflow and activate the session
import tensorflow as tf
sess = tf.InteractiveSession()

# create two placeholders for the data and the labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# create (and initialize) the variables for the weights and the biases
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# define the last layer and the loss function
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define how you want the 'network' to be trained, in this case we have chosen
# to minimize the cross entropy function using the gradient descent optimizer
# learning rate: 3e-4 - https://twitter.com/karpathy/status/801621764144971776
train_step = tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)

# 'initialize' the variables for the adam optimizer
sess.run(tf.initialize_all_variables())

# on each iteration, get a new batch of data and do the training on it
for i in range(3000):
  batch = mnist.train.next_batch(256)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# calculate and print the accuracy in the tesing set  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Training set accuracy ' + str(accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels})))
print('Testing set accuracy ' + str(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

# Why the 'network' is giving us these bad results?
# Answer: Spoiler Alert: the 'network' actually isn't a neural network at all,
# In fact, it is just a softmax regression algorithm.