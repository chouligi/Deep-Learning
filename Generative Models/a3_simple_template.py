from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [20,10]

import argparse
import pickle


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

#Code for digit plotting from lab assignment 2 in Machine Learning 1
def plot_digits(data, num_cols,index, targets=None, shape=(28,28)):
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



class NaiveBayesModel(object):

    def __init__(self, w_init, b_init = None, c_init = None):
        """
        :param w_init: An (n_categories, n_dim) array, where w[i, j] represents log p(X[j]=1 | Z[i]=1)
        :param b_init: A (n_categories, ) vector where b[i] represents log p(Z[i]=1), or None to fill with zeros
        :param c_init: A (n_dim, ) vector where b[j] represents log p(X[j]=1), or None to fill with zeros
        """
        #Initialize parameters
        self._w = w_init
        self._b = b_init
        self._c = c_init

    def log_p_x_given_z(self, x, z):
        """
        :param x: An (n_samples, n_dims) tensor
        :param z: An (n_samples, n_labels) tensor of integer class labels
        :return: An (n_samples, n_labels) tensor  p_x_given_z where result[i, j] indicates p(X=x[i] | Z=z[j])
        """
        #log_p_x_given_z
        #Create N x K x D array
        gather = tf.gather(self._w, z)
        
        #Form the Bernoulli distribution
        bern = tf.distributions.Bernoulli(logits = gather+self._c , name = 'Bernoulli')
        
        #logits
        #Reshape weights to fit dimensions
        log  = bern.log_prob(x[:,None,:],name='log_prob')
        
        #Sum along dimension with digits
        p_x_given_z = tf.reduce_sum(log,axis = 2)

        return p_x_given_z

    def log_p_x(self, x, p_x_given_z,z):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of log-probabilities assigned to each point
        """
        #log_p_x
        #Create categorical distribution for z
        self.p_z = tf.distributions.Categorical(logits = tf.nn.softmax(self._b))
        
        #Compute log_p(Z=k)
        z_prior = self.p_z.log_prob(z)
        
        #Add the two quantities and then use logsumexp to obtain log_p(x)
        summed = z_prior+p_x_given_z
        log_px = tf.reduce_logsumexp(summed,axis = 1)
        return log_px

    
    def sample(self, n_samples):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimenionality of your input
        """
        #Sample k from the categorical distribution of latent codes
        k = self.p_z.sample([n_samples])
        
        #Compute mean data activation per dimension
        gather_sample = tf.gather(self._w, k)
        
        #create the Bernoulli in order to sample
        bern_sample = tf.distributions.Bernoulli(logits = gather_sample + self._c , name = 'Bernoulli')
        #sample from the Bernoulli
        samples = bern_sample.sample()
        
        #Problem 6: plot expected pixel values given k
        
        #Create all K=20 categories
        categ = [i for i in range(0,20)]
        
        epv_gather = tf.gather(self._w, categ)
        
        #Expected pixel values to plot
        e_p_v = tf.nn.sigmoid(epv_gather + self._c)
        
        return samples, k, e_p_v

def train_simple_generative_model_on_mnist(n_categories=20, initial_mag = 0.01, optimizer='rmsprop', learning_rate=.01, n_epochs=20, test_every=100,minibatch_size=100, plot_n_samples=16,save_every=1500):
    """
    Train a simple Generative model on MNIST and plot the results.

    :param n_categories: Number of latent categories (K in assignment)
    :param initial_mag: Initial weight magnitude
    :param optimizer: The name of the optimizer to use
    :param learning_rate: Learning rate for the optimization
    :param n_epochs: Number of epochs to train for
    :param test_every: Test every X iterations
    :param minibatch_size: Number of samples in a minibatch
    :param plot_n_samples: Number of samples to plot
    """
    tf.reset_default_graph()

    # Get Data
    x_train, x_test = load_mnist_images(binarize=True)
    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    
    with tf.name_scope('input'):
        x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors

    # Build the model
    
    #initializers
    initializer_weights = tf.random_normal([n_categories, n_dims], mean = 0.0, stddev = initial_mag)
    initializer_biases  = tf.constant_initializer(0.0)
    
    #Create necessary variables
    with tf.variable_scope('weights'):
        w = tf.get_variable('weights', initializer= initializer_weights)
    #add to collection to reuse
    tf.add_to_collection('vars', w)

    with tf.variable_scope('bias'):
        b = tf.get_variable('biases', [n_categories], initializer= initializer_biases)
    tf.add_to_collection('vars', b)

    with tf.variable_scope('c_param'):
        c = tf.get_variable('c_params', [n_dims], initializer= initializer_biases)
    tf.add_to_collection('vars', c)

    #Create the Naive Bayes model
    model= NaiveBayesModel(w,b,c)

    #Randomly initialize the latent variable
    z = np.random.choice(n_categories,size = [minibatch_size,n_categories], p = None)
    with tf.name_scope('p_x_given_z'):
        p_x_given_z = model.log_p_x_given_z(x = x_minibatch, z = z)
    with tf.name_scope('p_x'):
        p_x = model.log_p_x(x_minibatch, p_x_given_z, z)
    with tf.name_scope('loss'):
        loss =  - tf.reduce_mean(p_x)
    tf.summary.scalar('loss',loss)

    inputs  = tf.placeholder(tf.int32,shape = [1000, n_dims],name = 'inputs')

    #Compute training and test log probabilities
    z1 = np.random.choice(n_categories,size = [1000,n_categories], p = None)
    with tf.name_scope('p_x_given_z_test'):
        p_x_given_z_test = model.log_p_x_given_z(x = inputs, z = z1)
    with tf.name_scope('p_x_test'):
        p_x_test = model.log_p_x(inputs, p_x_given_z_test, z1)
    with tf.name_scope('log_prob'):
        log_prob =   tf.reduce_mean(p_x_test)
    tf.summary.scalar('log_prob',log_prob)

    #Use the RMSProp optimizer
    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    with tf.name_scope('train'):
        train_step = optimizer.minimize(loss)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./summaries/' + '/train',graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter('./summaries/' + '/test')

    saver = tf.train.Saver()

    #Sample from the model
    with tf.name_scope('samples'):
        samples = model.sample(plot_n_samples)

    #Start the session
    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        n_steps = (n_epochs * n_samples)/minibatch_size
        for i in range(int(n_steps)):
            #TRAINING
            
            # Only for time measurement of step through network
            t1 = time.time()
            #Train step
            sess.run(train_step)
            # Only for time measurement of step through network
            t2 = time.time()
            examples_per_second = minibatch_size/float(t2-t1)
            
            
            #TESTING
            if i%test_every==0:
                l,summary = sess.run([loss,merged], feed_dict={inputs: x_test[0:1000]})

                test_writer.add_summary(summary,i)
                
                print("[{}] Train Step {:04}/{:04}, Batch Size = {}, ""Examples/Sec = {:.2f}, Loss = {}".format(datetime.now().strftime("%Y-%m-%d %H:%M"), i, int(n_steps),minibatch_size,examples_per_second, l ))

                sample_digits,classes , e_p_v = sess.run(samples)
                
                #Problem 6: plot images of expected pixel values given category
                
                #Uncomment to run the plotting
                #plot_digits(e_p_v, num_cols=4, index= i, targets = None)

                #Problem 7: plot samples from the trained model
                #Uncomment to run the plotting
                #plot_digits(sample_digits, num_cols=4, index= i, targets = classes)

                summary2 = sess.run(merged, feed_dict={inputs: x_train[0:1000]})
                train_writer.add_summary(summary2,i)
            
                #Save the digits
                
                #Uncomment to do the saving
                #f = open('samples_{}'.format(i), 'wb')
                #pickle.dump(sample_digits, f)
                #f.close()
            
            
            #Save the model variables
            if i%save_every==0:
                save_path = saver.save(sess, "./tmp/model.ckpt")
                print("Model saved in file: %s" % save_path)

    train_writer.close()
    test_writer.close()
    sess.close()

if __name__ == '__main__':
    train_simple_generative_model_on_mnist()
