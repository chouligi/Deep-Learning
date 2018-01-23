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

import os
import time
from datetime import datetime
import argparse

import numpy as np
import tensorflow as tf

#import utils
from dataset import TextDataset
from model import TextGenerationModel


def train(config):

    # Initialize the text dataset
    dataset = TextDataset(config.txt_file)

    # Initialize the model
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers,
        dropout_keep_prob = config.dropout_keep_prob
    )

    ###########################################################################
    # Implement code here.
    ###########################################################################
    
    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.int32, shape = [config.batch_size,config.seq_length], name = 'inputs')
        labels = tf.placeholder(tf.int32, shape=[config.batch_size,config.seq_length],name='labels')
        input_sample = tf.placeholder(tf.int32, shape = [config.batch_size,1],name = 'input_sample')
        state = tf.placeholder(tf.float32, [config.lstm_num_layers,2, config.batch_size, config.lstm_num_hidden])
    
    #Create tuple for the state placeholder
    layer = tf.unstack(state, axis=0)
    rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(layer[i][0], layer[i][1])
                         for i in range(config.lstm_num_layers)])

    #Logits
    with tf.name_scope('logits'):
        logits,_ = model._build_model(inputs,rnn_tuple_state)

    #Loss
    with tf.name_scope('loss'):
        loss = model._compute_loss(logits, labels)
    tf.summary.scalar('loss',loss)

    #Generate text
    with tf.name_scope('sample_logits'):
        sample_logits, final_state = model._build_model(input_sample,rnn_tuple_state)

    #predictions
    with tf.name_scope('predictions'):
        predictions = model.predictions(sample_logits)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    #decaying learning rate
    decaying_learning_rate = tf.train.exponential_decay(config.learning_rate, global_step,config.learning_rate_step , config.learning_rate_decay,name ='decaying_eta')
    tf.add_to_collection(tf.GraphKeys, decaying_learning_rate)

    # Define the optimizer
    optimizer = tf.train.RMSPropOptimizer(decaying_learning_rate)

    # Compute the gradients for each variable
    grads_and_vars = optimizer.compute_gradients(loss)
    #train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    grads, variables = zip(*grads_and_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=config.max_norm_gradient)
    apply_gradients_op = optimizer.apply_gradients(zip(grads_clipped, variables), global_step=global_step)

    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(config.summary_path + '/test',graph=tf.get_default_graph())

    #Initial zero state
    init_state = np.zeros((config.lstm_num_layers, 2, config.batch_size, config.lstm_num_hidden))


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    ###########################################################################
    # Implement code here.
    ###########################################################################

    for train_step in range(int(config.train_steps)):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################################
        # Implement code here.
        #######################################################################

        x_train, y_train = dataset.batch(config.batch_size,config.seq_length)
        #train
        sess.run(apply_gradients_op, feed_dict = {inputs: x_train, labels: y_train, state: init_state })


        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Output the training progress
        if train_step % config.print_every == 0:
            
            l, summary = sess.run([loss, merged], feed_dict= {inputs: x_train, labels: y_train, state: init_state})
            test_writer.add_summary(summary,train_step)
            
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), train_step+1,
                int(config.train_steps), config.batch_size, examples_per_second,l
            ))

        if train_step % config.sample_every == 0:
            sample_inputs = (np.random.randint(0, dataset.vocab_size,size = config.batch_size))
            new_sample= np.reshape(sample_inputs,(sample_inputs.shape[0],1))
            new_sentence = np.empty([(config.batch_size),(config.seq_length)])
            #Generate new sentence of length seq_length
            for i in range(config.seq_length):
                if i==0:
                    pred,final = sess.run([predictions,final_state], feed_dict = {input_sample: new_sample, state: init_state})
                    new_sample = pred.T
                    new_sentence[:,i][:,None] = new_sample
                #When unrolling for 30 timesteps, save the state and feed it again in the model
                elif (i >= 30 & i < 60):
                    pred,final = sess.run([predictions,final_state], feed_dict = {input_sample: new_sample, state: final})
                    new_sample = pred.T
                    new_sentence[:,i][:,None] = new_sample
                else:
                    pred,final = sess.run([predictions,final_state], feed_dict = {input_sample: new_sample, state: final})
                    new_sample = pred.T
                    new_sentence[:,i][:,None] = new_sample

            for idx, elem in enumerate(new_sentence):
                #We can skip the .encode('utf-8') for better looking output. It was used in order not to produce errors when running in surfsara.
                print('Sentence {}:{} {}'.format(idx,dataset.convert_to_string(sample_inputs)[idx].encode('utf-8'),dataset.convert_to_string(elem).encode('utf-8')))


    test_writer.close()
    sess.close()


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm_gradient', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5, help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False, help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
