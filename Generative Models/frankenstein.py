#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:07:52 2017

@author: George Chouliaras
"""

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


tf.reset_default_graph() 

#Function to plot the frankenstein digits
def plot_frankenstein(data, num_cols, targets=None, shape=(28,28)):
    plt.figure(figsize=(10, 8))
    num_digits = data.shape[0]
    num_rows = int(num_digits/num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i+1)
        cmap = plt.cm.gray
        plt.imshow(data[i].reshape(shape), interpolation='none', cmap=cmap)
        if targets is not None:
            plt.title('Log-probability: {0:.2f}'.format(targets[i]), fontsize = '8')
            #plt.title('Category: {}'.format(int(targets)))
        #plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle('Frankenstein digits',fontsize = '12')

    plt.show()
 
#Function to plot the unmodified digits
def plot_unmodified(data, num_cols, targets=None, shape=(28,28)):
    plt.figure(figsize=(10, 8))
    num_digits = data.shape[0]
    num_rows = int(num_digits/num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i+1)
        cmap = plt.cm.gray
        plt.imshow(data[i].reshape(shape), interpolation='none', cmap=cmap)
        if targets is not None:
            plt.title('Log-probability: {0:.2f}'.format(targets[i]), fontsize = '8')
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle('Unmodified digits',fontsize = '12')
    plt.show()


#import digits
f = open('./sampled_digits_NB/digit9', 'rb')
digit9 = pickle.load(f)
f.close()

f = open('./sampled_digits_NB/digit1', 'rb')
digit1 = pickle.load(f)
f.close()

f = open('./sampled_digits_NB/samples_10400', 'rb')
digits_10400 = pickle.load(f)
f.close()


f = open('./sampled_digits_NB/samples_10200', 'rb')
digits_10200 = pickle.load(f)
f.close()


f = open('./sampled_digits_NB/samples_8500', 'rb')
digits_8500 = pickle.load(f)
f.close()

f = open('./sampled_digits_NB/samples_7900', 'rb')
digits_7900 = pickle.load(f)
f.close()

f = open('./sampled_digits_NB/samples_7300', 'rb')
digits_7300 = pickle.load(f)
f.close()

f = open('./sampled_digits_NB/samples_7500', 'rb')
digits_7500 = pickle.load(f)
f.close()

f = open('./sampled_digits_NB/samples_4900', 'rb')
digits_4900 = pickle.load(f)
f.close()

#Save digits
digit0 = digits_10400[9]

digit2 = digits_10200[2]

digit6 = digits_8500[2]

digit7 = digits_8500[12]

digit8 = digits_4900[1]

digit5 = digits_7900[12]

digit1_straight=  digits_7900[9]

digit3 = digits_7300[9]

#Create Franeknstein digits

#FRANK1
#Merge 2 and 9

digit2_r = np.reshape(digit2,(28,28))
half2  = digit2_r[:,:14]

digit9_r = np.reshape(digit9,(28,28))
half9 = digit9_r[:,14:]
frank1 = np.concatenate((half2,half9), axis = 1)

#FRANK2 
#Merge 7 and 6

digit7_r = np.reshape(digit7,(28,28))
half7  = digit7_r[:,:14]

digit6_r = np.reshape(digit6,(28,28))
half6 = digit6_r[:,14:]

frank2 = np.concatenate((half7,half6), axis = 1)

#FRANK3
#Merge 1 straight and 8

digit1_straight_r = np.reshape(digit1_straight,(28,28))
half1_straight  = digit1_straight_r[:,14:]

digit8_r = np.reshape(digit8,(28,28))
half8 = digit8_r[:,:14]

frank3 = np.concatenate((half8,half1_straight), axis = 1)

#FRANK4
#Merge 5 and 2 

digit5_r = np.reshape(digit5,(28,28))
half5  = digit5_r[:,:14]

digit2_r = np.reshape(digit2,(28,28))
half2 = digit2_r[:,14:]

frank4 = np.concatenate((half5,half2), axis = 1)

#FRANK5
#Merge 7 and 3

digit3_r = np.reshape(digit3,(28,28))
half3  = digit3_r[:,14:]

digit7_r = np.reshape(digit7,(28,28))
half7 = digit7_r[:,:14]

frank5 = np.concatenate((half7,half3), axis = 1)

#FRANK6
#Merge 0 and 1

digit1_r = np.reshape(digit1,(28,28))
half1  = digit1_r[:,14:]

digit0_r = np.reshape(digit0,(28,28))
half0 = digit0_r[:,:14]

frank6 = np.concatenate((half0,half1), axis = 1)

#FRANK7
#Merge 6 and 9

digit6_r = np.reshape(digit6,(28,28))
half6  = digit6_r[:,:14]

digit9_r = np.reshape(digit9,(28,28))
half9 = digit9_r[:,14:]

frank7 = np.concatenate((half6,half9), axis = 1)

#FRANK8
#Merge 3 and 6

digit6_r = np.reshape(digit6,(28,28))
half6  = digit6_r[:,14:]

digit3_r = np.reshape(digit3,(28,28))
half3 = digit3_r[:,:14]

frank8 = np.concatenate((half3,half6), axis = 1)

#FRANK9 
#Merge 7 and 5

digit7_r = np.reshape(digit7,(28,28))
half7  = digit7_r[:,:14]

digit5_r = np.reshape(digit5,(28,28))
half5 = digit5_r[:,14:]

frank9 = np.concatenate((half7,half5), axis = 1)

#FRANK10
#Merge 0 and 2

digit0_r = np.reshape(digit0,(28,28))
half0  = digit0_r[:,:14]

digit2_r = np.reshape(digit2,(28,28))
half2 = digit2_r[:,14:]

frank10 = np.concatenate((half0,half2), axis = 1)

#Reshape all the frankenstein digits to put in an array

frank1_r = np.reshape(frank1, (784,))
frank2_r = np.reshape(frank2, (784,))
frank3_r = np.reshape(frank3, (784,))
frank4_r = np.reshape(frank4, (784,))
frank5_r = np.reshape(frank5, (784,))
frank6_r = np.reshape(frank6, (784,))
frank7_r = np.reshape(frank7, (784,))
frank8_r = np.reshape(frank8, (784,))
frank9_r = np.reshape(frank9, (784,))
frank10_r = np.reshape(frank10, (784,))

#Put all the digits in an array
frankenstein = np.vstack((frank1_r,frank2_r,frank3_r,frank4_r,frank5_r,frank6_r,frank7_r,frank8_r,frank9_r,frank10_r))

unmodified=np.vstack((digit0,digit1,digit1_straight,digit2,digit3,digit5,digit6,digit7,digit8,digit9))

#Parameters

n_categories=20
initial_mag = 0.01
optimizer='rmsprop'
learning_rate=.01
n_epochs=20
test_every=100
minibatch_size=100
plot_n_samples=16

#RESTORE THE MODEL TO SAMPLE

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./trained_NB/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./trained_NB'))

all_vars = tf.get_collection('vars')
#Save the weights to put them as initializations
w_saved = sess.run(all_vars[0])
b_saved = sess.run(all_vars[1])
c_saved = sess.run(all_vars[2])

sess.close()


#Find log probability of frankenstein and unmodified digits
import a3_simple_template


N = 10 




#Pass the saved weights and biases
with tf.variable_scope('weights'):
            w = tf.get_variable('weights', initializer= w_saved)
        
with tf.variable_scope('bias'):
            b = tf.get_variable('biases', initializer= b_saved)
        
with tf.variable_scope('c_param'):
            c = tf.get_variable('c_params', initializer= c_saved)

model= a3_simple_template.NaiveBayesModel(w,b,c)


z = np.random.choice(n_categories,size = [N,n_categories], p = None)

gather = tf.gather(w, z)
        
bern = tf.distributions.Bernoulli(logits = gather+c , name = 'Bernoulli')

#For unmodified
log_un  = bern.log_prob(unmodified[:,None,:],name='log_prob_unmodified')
        
p_x_given_z_un = tf.reduce_sum(log_un,axis = 2)

p_z_un = tf.distributions.Categorical(logits = b)
     
z_prior_un = p_z_un.log_prob(z)
        
#Add the two quantities and then use logsumexp to obtain log_p(x)
summed_un = z_prior_un+p_x_given_z_un
log_px_un = tf.reduce_logsumexp(summed_un,axis = 1)

#For fraknenstein

log_fr  = bern.log_prob(frankenstein[:,None,:],name='log_prob_frankenstein')
        
p_x_given_z_fr = tf.reduce_sum(log_fr,axis = 2)

p_z_fr = tf.distributions.Categorical(logits = b)
     
z_prior_fr = p_z_fr.log_prob(z)
        
#Add the two quantities and then use logsumexp to obtain log_p(x)
summed_fr = z_prior_fr+p_x_given_z_fr
log_px_fr = tf.reduce_logsumexp(summed_fr,axis = 1)


#Plot frankenstein digits


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
px_un = sess.run(log_px_un)
px_fr = sess.run(log_px_fr)

#Do the plotting
plot_frankenstein(frankenstein, num_cols=5, targets = px_fr)
plot_unmodified(unmodified, num_cols=5, targets = px_un)



sess.close()
