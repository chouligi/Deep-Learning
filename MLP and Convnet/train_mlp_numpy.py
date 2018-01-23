"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import cifar10_utils
import mlp_numpy

# Default constants
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DNN_HIDDEN_UNITS_DEFAULT = '100'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

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
  Performs training and evaluation of MLP model. Evaluate your model on the whole test set each 100 iterations.
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

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

  n_classes = y_test.shape[1]
  dim = x_test.shape[1]*x_test.shape[2]*x_test.shape[3]
  mlp = mlp_numpy.MLP(dnn_hidden_units, n_classes, weight_decay=FLAGS.weight_reg_strength, weight_scale=FLAGS.weight_init_scale)

  #reshape test set
  x_test_res  = np.reshape(x_test, (x_test.shape[0], dim))
  #Perform SGD
  for i in range(FLAGS.max_steps+1):
    batch_x,batch_y = next_batch(FLAGS.batch_size,x_train,y_train)
    x_res= np.reshape(batch_x, (batch_x.shape[0], batch_x.shape[1]*batch_x.shape[2]*batch_x.shape[3]))
    logits = mlp.inference(x_res)
    loss = mlp.loss(logits,batch_y)
    mlp.train_step(loss,FLAGS.learning_rate)
    if i in [i*100 for i in range(1,16)]:
        print('Performing iteration ' + str(i) + ' ...')
        x_test_res = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))
        #predict
        preds = mlp.inference(x_test_res)
        acc = mlp.accuracy(preds,y_test)
        print('Accuracy on the test set: ' + str(acc*100) + '%')
    ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
      os.makedirs(FLAGS.data_dir)

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
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
