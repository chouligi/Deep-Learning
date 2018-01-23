# MIT License
# 
# Copyright (c) 2017 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-10-19

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LSTM(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.contrib.layers.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)

        # Initialize the stuff you need
        #Initialize weights
        with tf.variable_scope('input_weights'):
            self._Wx = tf.get_variable('W_input', [4, self._num_classes, self._num_hidden], initializer= initializer_weights)


        with tf.variable_scope('hidden_weights'):
           self._Wh = tf.get_variable('W_hidden', [4,self._num_hidden,self._num_hidden], initializer= initializer_weights)
           self._b = tf.get_variable('biases', [4,self._num_hidden], initializer = initializer_biases)


        with tf.variable_scope('output_weights'):
            self._Wo = tf.get_variable('W_output', [self._num_hidden,self._num_classes], initializer= initializer_weights)
            self._bo = tf.get_variable('b_output', [self._num_classes], initializer= initializer_biases)

        #2 initial states: for cell state ct and memory state ht
        self._initial_state = tf.zeros([2,self._batch_size,self._num_hidden])
            
    def _lstm_step(self, lstm_state_tuple, x):
        # Single step through LSTM cell ...
        #Get cell state ct and memory state ht
        ht, ct = tf.unstack(lstm_state_tuple)
        
        #input modulation gate / gate weights
        g = tf.tanh(tf.matmul(x,self._Wx[0]) + tf.matmul(ht,self._Wh[0]) + self._b[0])
        #input gate
        i = tf.sigmoid(tf.matmul(x,self._Wx[1]) + tf.matmul(ht,self._Wh[1]) + self._b[1])
        #forget gate
        f = tf.sigmoid(tf.matmul(x,self._Wx[2]) + tf.matmul(ht,self._Wh[2]) + self._b[2])
        #output gate
        o = tf.sigmoid(tf.matmul(x,self._Wx[3]) + tf.matmul(ht,self._Wh[3]) + self._b[3])
        
        #new internal cell state
        ct_new = g*i + ct*f
        
        #output state
        ht_new = tf.tanh(ct_new)*o
        return tf.stack([ht_new,ct_new])
    
    def get_output(self,ht):
        p = tf.matmul(ht,self._Wo) + self._bo
        return p
    
    def compute_logits(self,x):
        # Implement the logits for predicting the last digit in the palindrome
        
        #Turn the inputs to one hot vectors
        inputs_oh = tf.one_hot(x, depth=self._num_classes)
        #Transpose in order to insert in tf.scan
        lstm_inputs = tf.transpose(inputs_oh,[1,0,2])
  
        states = tf.scan(self._lstm_step, lstm_inputs, initializer = self._initial_state)
     
        #Get the outputs for all the states, we will use only the last one
        #We insert the ht (states[:,0,:,:])
        all_p = tf.map_fn(self.get_output, states[:,0,:,:])
    

        logits = all_p[-1]
        return logits

    def compute_loss(self,logits,labels):
        # Implement the cross-entropy loss for classification of the last digit
        #create one hot labels
        labels_oh  = tf.one_hot(indices = labels, depth = self._num_classes)
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels_oh, logits = logits)
        
        loss = tf.reduce_mean(cross_entropy, 0)
        
        return loss

    def accuracy(self,logits,labels):
        # Implement the accuracy of predicting the
        # last digit over the current batch ...
        labels_oh  = tf.one_hot(indices = labels, depth = self._num_classes)
        correct_prediction = tf.equal(tf.argmax(labels_oh,1), tf.argmax(logits,1))
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
        return accuracy
