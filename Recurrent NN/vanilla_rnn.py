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

import numpy as np
import tensorflow as tf

################################################################################

class VanillaRNN(object):
    
    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):
        
        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size
        
        
        
        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)
        
        # Initialize the stuff you need
        
        #Initialize the weights
        with tf.variable_scope('input_projection'):
            self._Whx = tf.get_variable('W_input', [self._num_classes, self._num_hidden], initializer= initializer_weights)
        
        with tf.variable_scope('hidden_weights'):
            self._Whh = tf.get_variable('W_hidden', [self._num_hidden,self._num_hidden], initializer= initializer_weights)
            self._bh = tf.get_variable('b_hidden', [self._num_hidden], initializer = initializer_biases)
        
        with tf.variable_scope('output_weights'):
            self._Who = tf.get_variable('W_output', [self._num_hidden,self._num_classes], initializer= initializer_weights)
            self._bo = tf.get_variable('b_output', [self._num_classes], initializer= initializer_biases)
        
        #initialize state
        self._initial_state = tf.zeros([self._batch_size,self._num_hidden])



    def _rnn_step(self, h_prev, x):
        # Single step through Vanilla RNN cell ...
        state = tf.tanh(x@self._Whx + h_prev@self._Whh + self._bh)
        return state
    
    def get_output(self,hidden_state):
        output = hidden_state@self._Who + self._bo
        return output
    
    
    def compute_logits(self,x):
        # Implement the logits for predicting the last digit in the palindrome
        
        #Turn the input to one hot vectors
        inputs_oh = tf.one_hot(x, depth=self._num_classes)
        
        all_hidden_states = tf.scan(self._rnn_step,
                                    tf.transpose(inputs_oh,[1,0,2]),
                                    initializer=self._initial_state, name = 'states')
            
        #Get the outputs for all the states, we will use only the last one
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
                                    
        logits = all_outputs[-1]
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

