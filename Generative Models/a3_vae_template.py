import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Lambda
from keras.models import Model
from keras import backend as K

import time
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

plt.rcParams["figure.figsize"] = [20,10]


def plot_digits(data, num_cols,idx, targets=None, shape=(28,28)):
    num_digits = data.shape[0]
    num_rows = int(num_digits/num_cols)
    plt.figure(idx)
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


class VariationalAutoencoder(object):

    def lower_bound(self, x, decoder_mean, mean_z_value,var_z_value):
        """
        :param x: A (n_samples, n_dim) array of data points
        :return: A (n_samples, ) array of the lower-bound on the log-probability of each data point
        """
        #Create bernoulli distribution
        bern = tf.distributions.Bernoulli(logits = decoder_mean , name = 'Bernoulli')
        log_px_given_z = bern.log_prob(x, name='log_prob')
    
        #Compute reconstruction error
        l_recon = tf.reduce_sum(log_px_given_z, axis = 1)

        #Compute regularization error = KL divergence

        #formula from: http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
        l_reg = -1/2 * tf.reduce_sum(1 + tf.log(var_z_value**2) - mean_z_value**2 - var_z_value**2, axis = 1)

        #Compute lower bound
        lb = l_recon - l_reg
        
        return lb


    def mean_x_given_z(self, z, mean_NN):
        """
        :param z: A (n_samples, n_dim_z) tensor containing a set of latent data points (n_samples, n_dim_z)
        :return: A (n_samples, n_dim_x) tensor containing the mean of p(X|Z=z) for each of the given points
        """
        #N.N that computes the mean for the bernoulli
        decoder_mean = mean_NN(z)
        
        return decoder_mean

    def sample(self, n_samples, z,mean_NN):
        """
        :param n_samples: Generate N samples from your model
        :return: A (n_samples, n_dim) array where n_dim is the dimensionality of your input
        """
        
        # build a digit generator that can sample from the learned distribution
        decoded_mean = self.mean_x_given_z(z, mean_NN)
        
        bern_sample = tf.distributions.Bernoulli(logits = decoded_mean , name = 'Bernoulli_sample')

        #get array of sampled digits
        digits = bern_sample.sample()
        
        return digits




def train_vae_on_mnist(z_dim=2, kernel_initializer='glorot_uniform', optimizer = 'adam',  learning_rate=0.001, n_epochs=4000,
        test_every=100, minibatch_size=100, encoder_hidden_sizes=[200, 200], decoder_hidden_sizes=[200, 200],
        hidden_activation='relu', plot_grid_size=10, plot_n_samples = 20, save_every= 1500, plot_every = 1000):
    """
    Train a variational autoencoder on MNIST and plot the results.

    :param z_dim: The dimensionality of the latent space.
    :param kernel_initializer: How to initialize the weight matrices (see tf.keras.layers.Dense)
    :param optimizer: The optimizer to use
    :param learning_rate: The learning rate for the optimizer
    :param n_epochs: Number of epochs to train
    :param test_every: Test every X training iterations
    :param minibatch_size: Number of samples per minibatch
    :param encoder_hidden_sizes: Sizes of hidden layers in encoder
    :param decoder_hidden_sizes: Sizes of hidden layers in decoder
    :param hidden_activation: Activation to use for hidden layers of encoder/decoder.
    :param plot_grid_size: Number of rows, columns to use to make grid-plot of images corresponding to latent Z-points
    :param plot_n_samples: Number of samples to draw when plotting samples from model.
    """
    
    def z_sampling(args):
        z_mean, z_var = args
        epsilon = tf.random_normal(shape =(tf.shape(z_mean)[0],z_dim), mean = 0., stddev = 1.0)
    
        return z_mean + z_var*epsilon

    # Get Data
    x_train, x_test = load_mnist_images(binarize=True)
    train_iterator = tf.data.Dataset.from_tensor_slices(x_train).repeat().batch(minibatch_size).make_initializable_iterator()
    n_samples, n_dims = x_train.shape
    x_minibatch = train_iterator.get_next()  # Get symbolic data, target tensors



    # Build Model
    
    #Build the encoder Q(z|X)
    hq = Sequential()
    #add 1st layer
    hq.add(Dense(units = encoder_hidden_sizes[0], input_dim = 784,kernel_initializer=kernel_initializer))
    hq.add(Activation(hidden_activation))
    #Add 2nd layer
    hq.add(Dense(units = encoder_hidden_sizes[1],kernel_initializer=kernel_initializer))
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
    h.add(Dense(decoder_hidden_sizes[0],input_dim = z_dim,kernel_initializer=kernel_initializer))
    h.add(Activation(hidden_activation))
    #2nd layer
    h.add(Dense(units = decoder_hidden_sizes[1],kernel_initializer=kernel_initializer))
    h.add(Activation(hidden_activation))

    #N.N that computes the mean for the bernoulli
    mean_NN = Sequential([h])
    mean_NN.add(Dense(784))

    #Build the model
    
    vae = VariationalAutoencoder()
    
    #Compute decoder_mean
    decoder_mean = vae.mean_x_given_z(z,mean_NN)
    
    #Compute lower bound
    with tf.name_scope('lower_bound'):
        lb = vae.lower_bound(x_minibatch, decoder_mean,mean_z_value,var_z_value)
        mean_lb = tf.reduce_mean(lb)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-lb)

    #Compute loss for training and test set
    inputs  = tf.placeholder(tf.float32,shape = [None, n_dims],name = 'inputs')
    mean_z_test = z_mean(inputs)
    var_z_test = z_var(inputs)

    z_test = Lambda(z_sampling, output_shape=(z_dim,))([mean_z_test,var_z_test])
    decoder_mean_test = vae.mean_x_given_z(z_test,mean_NN)

    with tf.name_scope('test_lower_bound'):
        test_lb = vae.lower_bound(inputs, decoder_mean_test,mean_z_test,var_z_test)
        mean_test_lb = tf.reduce_mean(test_lb)
    with tf.name_scope('test_loss'):
        loss_test = tf.reduce_mean(-test_lb)
    tf.summary.scalar('lower_bound',mean_test_lb)


    with tf.name_scope('z_sample'):
        sample_z = tf.placeholder(tf.float32, shape=[plot_n_samples, z_dim])
    #Sampling
    with tf.name_scope('Samples'):
        sampled_digits = vae.sample(plot_n_samples, sample_z, mean_NN)

    #Training step

    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope("Adam_optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./summaries_VAE/' + '/train_VAE',graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter('./summaries_VAE/' + '/test_VAE')

    with tf.Session() as sess:
        print('Session Starts ...')
        
        sess.run(train_iterator.initializer)  # Initialize the variables of the data-loader.

        sess.run(tf.global_variables_initializer())  # Initialize the model parameters.
        n_steps = (n_epochs * n_samples)/minibatch_size
        for i in range(int(n_steps)):
            if i%test_every==0:
                #RUN TEST AND RECORD LOG-PROB PERIODICALLY
                t1 = time.time()
                #test_loss
                l, summary = sess.run([loss,merged], feed_dict={inputs: x_test})
                
                #train_loss
                summary2 = sess.run(merged, feed_dict={inputs: x_train})
                
                t2 = time.time()
                examples_per_second = 100/float(t2-t1)
                train_writer.add_summary(summary2,i)
                test_writer.add_summary(summary,i)
                print("[{}] Train Step {:04}/{:04}, Batch Size = {}, ""Examples/Sec = {:.2f}, Loss = {}".format(datetime.now().strftime("%Y-%m-%d %H:%M"), i, int(n_steps),minibatch_size,examples_per_second, l ))
            
            #Plot Digits
            if i%plot_every ==0:
                samples = sess.run(sampled_digits,feed_dict = {sample_z:np.random.normal(0.0,1.0,(plot_n_samples,z_dim)) })
                
                #NOTE: While training you have to close the figure to continue training!
                plot_digits(samples, num_cols=4,idx = i)
                
                #Uncomment to save the digits
                #f = open('samples_VAE_{}'.format(i), 'wb')
                #pickle.dump(samples, f)
                #f.close()
            
            
            #Uncomment to save the model weights
            #if i%save_every==0:
            #    z_mean.save_weights('z_mean_weights.h5')
            #    z_var.save_weights('z_var_weights.h5')
            #    mean_NN.save_weights('mean_NN_weights.h5')
            #   hq.save_weights('hq_weights.h5')
            #   h.save_weights('h_weights.h5')
                
            #   print('Model weights saved')
            
            

            #CALL TRAINING FUNCTION HERE
            sess.run(train_op)

    test_writer.close()
    train_writer.close()
    sess.close()


if __name__ == '__main__':
    train_vae_on_mnist()
