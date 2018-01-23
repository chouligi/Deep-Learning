#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:10:58 2017

@author: George Chouliaras
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_mnist_images(binarize=True):
    """
    :param binarize: Turn the images into binary vectors
    :return: x_train, x_test  Where
        x_train is a (55000 x 784) tensor of training images
        x_test is a  (10000 x 784) tensor of test images
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
    x_train = mnist.train.images
    x_test = mnist.test.images
    if binarize:
        x_train = (x_train>0.5).astype(x_train.dtype)
        x_test = (x_test>0.5).astype(x_test.dtype)
    return x_train, x_test

def plot_digits(data, num_cols, targets=None, shape=(28,28)):
    num_digits = data.shape[0]
    num_rows = int(num_digits/num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i+1)
        cmap = plt.cm.gray
        plt.imshow(data[i].reshape(shape), interpolation='none', cmap=cmap)
        if targets is not None:
            plt.title('Category: {}'.format(int(targets[i])))
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
    
#Create and train model here 

from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda



def z_sampling(args):
    #epsilon = tf.random_normal(shape =(1,z_dim), mean = 0., stddev = 1.0)
    z_mean, z_var = args
    epsilon = tf.random_normal(shape =(tf.shape(z_mean)[0],z_dim), mean = 0., stddev = 1.0)
    
    return z_mean + z_var*epsilon

z_dim=2
kernel_initializer='glorot_uniform'
optimizer = 'adam'
learning_rate=0.001
n_epochs=4000
test_every=100
minibatch_size=100
encoder_hidden_sizes=[200, 200]
decoder_hidden_sizes=[200, 200]
hidden_activation='relu'
plot_grid_size=10
plot_n_samples = 20


# Get Data
x_train, x_test = load_mnist_images(binarize=True)
train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
n_samples, n_dims = x_train.shape
x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors


#Build the encoder Q(z|X)
hq = Sequential()
    #add 1st layer
hq.add(Dense(units = encoder_hidden_sizes[0], input_dim = 784))
hq.add(Activation(hidden_activation))
    #Add 2nd layer
hq.add(Dense(units = encoder_hidden_sizes[1]))
hq.add(Activation(hidden_activation))


#mu_z
z_mean = Sequential([hq])
z_mean.add(Dense(z_dim))

    
    
    #Sigma_z
z_var = Sequential([hq])
z_var.add(Dense(z_dim))
z_var.add(Activation(tf.exp))

    
    #Evaluate mean_z and sigma_z
mean_z_value = z_mean(x_minibatch)
var_z_value = z_var(x_minibatch)

    #Take z
z = Lambda(z_sampling, output_shape=(z_dim,))([mean_z_value,var_z_value])
    
    #Decoder p(x|z)

h = Sequential()
    #1st layer
h.add(Dense(decoder_hidden_sizes[0],input_dim = z_dim))
h.add(Activation(hidden_activation))
    #2nd layer
h.add(Dense(units = decoder_hidden_sizes[1]))
h.add(Activation(hidden_activation))


mean_NN = Sequential([h])
mean_NN.add(Dense(784))




n = 15  # figure with 15x15 digits
digit_size = 28

nx = ny = 20

#We vary z between -2,2 to get a high quality representation
grid_x = np.linspace(-2,2,nx)
grid_y = np.linspace(-2,2,ny)

mesh= np.empty((digit_size*nx,digit_size*ny ))


sample_z = tf.placeholder(tf.float32, shape=[1, z_dim])
decoded_mean = mean_NN(sample_z)

#apply sigmoid to turn into probabilities
x_decoded = tf.nn.sigmoid(decoded_mean)

with tf.Session() as sess:
    sess.run(train_iterator.initializer)  # Initialize the variables of the data-loader.

    sess.run(tf.global_variables_initializer())  # Init
    
    #load the weights from the trained model
    h.load_weights('h_weights_20.h5')
    mean_NN.load_weights('mean_NN_weights_20.h5')
    
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])     
    
            x_mean =  sess.run(x_decoded, feed_dict= {sample_z: z_sample})
        
            mesh[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(digit_size,digit_size)

    plt.figure(figsize=(10, 10))
    plt.imshow(mesh, cmap='Greys_r')
    plt.tight_layout()
    #plt.savefig('VAE_manifold.png',dpi = 300)
    plt.axis('off')
    plt.show()
    
sess.close()
