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

class TextGenerationModel(object):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden, lstm_num_layers, dropout_keep_prob):

        self._seq_length = seq_length
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._vocab_size = vocabulary_size
        self._dropout_keep_prob = dropout_keep_prob

        # Initialization:
        initializer_weights = tf.contrib.layers.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)
    
        with tf.variable_scope('LSTM'):
            #the LSTM
            self.lstm_cells = [tf.contrib.rnn.LSTMCell(self._lstm_num_hidden, state_is_tuple=True) for i in range(self._lstm_num_layers) ]
            self._lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=True)
            self._lstm = tf.contrib.rnn.DropoutWrapper(self._lstm, input_keep_prob=self._dropout_keep_prob)
        with tf.variable_scope('output_weights'):
                self._Wo = tf.get_variable('W_output', [self._lstm_num_hidden,self._vocab_size], initializer= initializer_weights)
                self._bo = tf.get_variable('b_output', [self._vocab_size], initializer= initializer_biases)
    
    def get_output(self,ht):
        p = tf.add(tf.matmul(ht,self._Wo) , self._bo)
        return p
    
    
    def _build_model(self,x,init_state):
        # Implement your model to return the logits per step of shape:
        #   [timesteps, batch_size, vocab_size]
        
        #make the inputs one hot
        rnn_inputs = tf.one_hot(x, depth = self._vocab_size)

        #unfold the network
        rnn_outputs, final_state = tf.nn.dynamic_rnn(self._lstm, rnn_inputs, initial_state=init_state)

        logits_per_step = tf.map_fn(self.get_output, tf.transpose(rnn_outputs,[1,0,2]))
        return logits_per_step, final_state

    def _compute_loss(self,logits,labels):
        # Cross-entropy loss, averaged over timestep and batch
        labels_oh  = tf.one_hot(indices = labels, depth = self._vocab_size)
        #We turn the labels to the same form as the logits (that's why the transpose)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf.transpose(labels_oh,[1,0,2]), logits = logits)
        #average across time step
        loss_avg = tf.reduce_mean(cross_entropy, axis= 0)
        #average across batch
        loss = tf.reduce_mean(loss_avg)
        
        return loss

    def probabilities(self,logits):
        # Returns the normalized per-step probabilities
        probabilities = tf.nn.softmax(logits)
        return probabilities

    def predictions(self,logits):
        # Returns the per-step predictions
        predictions = tf.argmax(logits,2)
        return predictions
