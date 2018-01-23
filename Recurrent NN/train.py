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
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils
from vanilla_rnn import VanillaRNN
from lstm import LSTM

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Setup the model that we are going to use
    if config.model_type == 'RNN':
        print("Initializing Vanilla RNN model...")
        model = VanillaRNN(
            config.input_length, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )
    else:
        print("Initializing LSTM model...")
        model = LSTM(
            config.input_length, config.input_dim, config.num_hidden,
            config.num_classes, config.batch_size
        )

    ###########################################################################
    # Implement code here.
    ###########################################################################
    
    #Create placeholders
    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.int32, shape = [config.batch_size, config.input_length -1], name = 'inputs')
        labels = tf.placeholder(tf.int32, shape=[config.batch_size],name='labels')
        test_inputs = tf.placeholder(tf.int32, shape = [config.batch_size, config.input_length -1], name = 'test_inputs')
        test_labels = tf.placeholder(tf.int32, shape=[config.batch_size],name='test_labels')
    
    #Compute the logits
    with tf.name_scope('logits'):
        logits = model.compute_logits(inputs)
    #Compute the loss
    with tf.name_scope('loss'):
        loss = model.compute_loss(logits, labels)
    tf.summary.scalar('loss',loss)
    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate)


    ###########################################################################
    # Implement code here.
    ###########################################################################

    ###########################################################################
    # QUESTION: what happens here and why? ->put threshold in order to avoid exploding gradients (gradient clipping)
    ###########################################################################
    global_step = tf.Variable(0, trainable=False, name='global_step')

    dummy = loss
    grads_and_vars = optimizer.compute_gradients(dummy)

    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)


    #Compute the accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('predictions'):
            predictions = model.compute_logits(test_inputs)
        with tf.name_scope('accuracy'):
            accuracy= model.accuracy(predictions,test_labels)
    tf.summary.scalar('accuracy',accuracy)


    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(config.summary_path + '/test',graph=tf.get_default_graph())


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    ############################################################################

    ###########################################################################
    # Implement code here.
    ###########################################################################

    for train_step in range(config.train_steps+1):

        # Only for time measurement of step through network
        t1 = time.time()
        
        
        batch = utils.generate_palindrome_batch(config.batch_size,config.input_length)
        #Take the first T-1 digits as input
        batch_x = batch[:,0:(config.input_length-1)]
        #Take the last digit as the label, correct class
        batch_y = batch[:,-1]
        sess.run(apply_gradients_op, feed_dict={inputs: batch_x,labels:batch_y})

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Print the training progress
        if train_step % config.print_every == 0:
            #Create batch to test
            batch_test = utils.generate_palindrome_batch(config.batch_size,config.input_length)
            batch_x_test = batch_test[:,0:(config.input_length-1)]
            batch_y_test = batch_test[:,-1]

            l,acc,summary = sess.run([loss,accuracy,merged], feed_dict={inputs: batch_x, labels: batch_y, test_inputs: batch_x_test,test_labels:batch_y_test})
            
            test_writer.add_summary(summary,train_step)
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, "
                  "Examples/Sec = {:.2f}, Accuracy = {}%, Loss = {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step,
                config.train_steps, config.batch_size, examples_per_second, acc, l
            ))
    test_writer.close()
    sess.close()


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2500, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=10.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')

    config = parser.parse_args()

    # Train the model
    train(config)




