# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:40:13 2016

@author: revan
"""

""" On this script, we are going to build a feedforward, fully connected neural
network, and then train and test it on the MNIST dataset """

# get the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# import tensorflow and activate the session
import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)
  
def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)  
  
# create two placeholders for the data and the labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# create and initialize weights and biases
W1 = weight_variable([784, 500])
b1 = bias_variable([500])
W2 = weight_variable([500, 500])
b2 = bias_variable([500])
W3 = weight_variable([500, 10])
b3 = bias_variable([10])

# compute the values of the first layer (using relu as activation function)
layer_1 = tf.add(tf.matmul(x, W1), b1)
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
layer_2 = tf.nn.relu(layer_2)

# compute the values of the output layer
y = tf.nn.softmax(tf.matmul(layer_2, W3) + b3)

# define the loss function, and the optimation algorithm
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# define the correct prediction, and the accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define the parameters for the dropout algorithm
keep_prob = tf.placeholder(tf.float32)
layer_1_drop = tf.nn.dropout(layer_1, keep_prob)

# initialize the variables and run the session
sess.run(tf.initialize_all_variables())

for i in range(3000):
  batch = mnist.train.next_batch(256)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))   