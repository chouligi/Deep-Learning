"""
This module implements training and evaluation of a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer
import numpy as np
import cifar10_utils
import mlp_tf

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.
DNN_HIDDEN_UNITS_DEFAULT = '100'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/cifar10'


# This is the list of options for command line arguments specified below using argparse.
# Make sure that all these options are available so we can automatically test your code
# through command line arguments.

# You can check the TensorFlow API at
# https://www.tensorflow.org/programmers_guide/variables
# https://www.tensorflow.org/api_guides/python/contrib.layers#Initializers
WEIGHT_INITIALIZATION_DICT = {'xavier': xavier_initializer(), # Xavier initialisation
                              'normal': tf.random_normal_initializer(mean = 0 ,stddev = 1e-4, dtype = tf.float32), # Initialization from a standard normal
                              'uniform': tf.random_uniform_initializer(minval = -1/np.sqrt(3072), maxval = 1/np.sqrt(3072)) , # Initialization from a uniform distribution
                             }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/contrib.layers#Regularizers
WEIGHT_REGULARIZER_DICT = {'none': None, # No regularization
                           'l1': l1_regularizer, # L1 regularization
                           'l2': l2_regularizer # L2 regularization
                          }

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/nn
ACTIVATION_DICT = {'relu': tf.nn.relu, # ReLU
                   'elu': tf.nn.elu, # ELU
                   'tanh': tf.nn.tanh, #Tanh
                   'sigmoid': tf.nn.sigmoid} #Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/api_guides/python/train
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer , # Adadelta
                  'adagrad':tf.train.AdagradOptimizer , # Adagrad
                  'adam':tf.train.AdamOptimizer , # Adam
                  'rmsprop': tf.train.RMSPropOptimizer  # RMSprop
                  }

FLAGS = None

def next_batch(num, x, y):
    #get indices of x
    idx = np.arange(0 , len(x))
    #shuffle the indices
    np.random.shuffle(idx)
    idx = idx[:num]
    shuffle_x = np.asarray([x[i] for i in idx])
    shuffle_y = np.asarray([y[i] for i in idx])
    
    return shuffle_x, shuffle_y

def train():
  #In order to run this function from Jupyter, we must add a class FLAGS as input argument for train
  """
  Performs training and evaluation of MLP model. Evaluate your model each 100 iterations
  as you did in the task 1 of this assignment. 
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  tf.set_random_seed(42)
  np.random.seed(42)
  tf.reset_default_graph()


  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  x_train, y_train  = cifar10.train.images, cifar10.train.labels
  x_test,y_test = cifar10.test.images, cifar10.test.labels
  #reshape x_test
  x_test_res  = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))

  n_classes = y_test.shape[1]
  dim = x_test.shape[1]*x_test.shape[2]*x_test.shape[3]

  if (FLAGS.weight_reg == 'l2' ):
        weight_reg = WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg](FLAGS.weight_reg_strength)
  elif (FLAGS.weight_reg == 'l1' ):
        weight_reg = WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg](FLAGS.weight_reg_strength)
  else:
        weight_reg = WEIGHT_REGULARIZER_DICT[FLAGS.weight_reg]

  mlp = mlp_tf.MLP(dnn_hidden_units,n_classes, is_training = None,weight_decay = FLAGS.weight_reg_strength,activation_fn = ACTIVATION_DICT[FLAGS.activation], weight_initializer = WEIGHT_INITIALIZATION_DICT[FLAGS.weight_init],weight_regularizer = weight_reg ,inp_dim = 3072)

  # Tensorflow Graph input
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 3072], name="x-data")
    y = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, n_classes], name="y-labels")
    test_x = tf.placeholder(tf.float32, shape = x_test_res.shape, name ='x-test')
    test_y = tf.placeholder(tf.int32, shape= y_test.shape,name = 'y-test')
  #forward pass
  logits = mlp.inference(x)
  #obtain loss
  with tf.name_scope('loss'):
    loss = mlp.loss(logits, y)
  tf.summary.scalar('loss',loss)
  #backward pass
  eta = FLAGS.learning_rate

  #for Adam optimizer
  beta1 = 0.91
  beta2 = 0.999
  epsilon = 0.0001

 #for RMSProp optimizer
  decay = 0.1
  momentum = 0.2
  epsilonR = 1.1

  with tf.name_scope('train'):
    if (FLAGS.optimizer == 'adam'):
        optimizer = OPTIMIZER_DICT[FLAGS.optimizer](eta,beta1,beta2,epsilon)
    elif (FLAGS.optimizer == 'rmsprop'):
        optimizer = OPTIMIZER_DICT[FLAGS.optimizer](eta,decay,momentum,epsilonR)
    else:
        optimizer = OPTIMIZER_DICT[FLAGS.optimizer](eta)
    train_step = mlp.train_step(loss, optimizer)

  with tf.name_scope('accuracy'):
    #compute output, the predictions
    preds = mlp.inference(test_x)
    #compute accuracy
    acc = mlp.accuracy(preds,test_y)
  tf.summary.scalar('accuracy',acc)

  #compute confusion matrix
  with tf.name_scope('confusion'):
    predictions = tf.argmax(preds,axis = 1)
    labels = tf.argmax(test_y,axis = 1)
    with tf.name_scope('confusion'):
        conf_matrix = tf.contrib.metrics.confusion_matrix(labels = labels, predictions = predictions, name='confusion_matrix')

  # Merge all the summaries and write them out to logdir
  merged = tf.summary.merge_all()

  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',graph=tf.get_default_graph())
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

  init = tf.global_variables_initializer()
  sess = tf.Session()
  #initialize
  sess.run(init)

  #Training
  for i in range(FLAGS.max_steps+1):
    batch_x,batch_y = next_batch(FLAGS.batch_size,x_train,y_train)
    x_res = np.reshape(batch_x, (batch_x.shape[0], batch_x.shape[1]*batch_x.shape[2]*batch_x.shape[3]))
    sess.run(train_step, feed_dict={x: x_res, y:batch_y})
    if (i % 100 ==0):
        summary,l,accur = sess.run([merged,loss,acc], feed_dict = {x: x_res, y:batch_y, test_x: x_test_res, test_y: y_test})
        print('Loss at step {}: {}'.format(i,l))
        print('Test accuracy at step {}: {}'.format(i,accur))
        test_writer.add_summary(summary,i)
    if (i == FLAGS.max_steps):
        conf = sess.run(conf_matrix, feed_dict = {test_x: x_test_res, test_y: y_test})
        print('Confusion matrix:')
        print(conf)
  train_writer.close()
  test_writer.close()
  sess.close()
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  # Make directories if they do not exists yet
  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--weight_init', type = str, default = WEIGHT_INITIALIZATION_DEFAULT,
                      help='Weight initialization type [xavier, normal, uniform].')
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg', type = str, default = WEIGHT_REGULARIZER_DEFAULT,
                      help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='Dropout rate.')
  parser.add_argument('--activation', type = str, default = ACTIVATION_DEFAULT,
                      help='Activation function [relu, elu, tanh, sigmoid].')
  parser.add_argument('--optimizer', type = str, default = OPTIMIZER_DEFAULT,
                      help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
