from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import convnet_tf
import cifar10_utils


LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

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
  Performs training and evaluation of ConvNet model.

  First define your graph using class ConvNet and its methods. Then define
  necessary operations such as savers and summarizers. Finally, initialize
  your model within a tf.Session and do the training.

  ---------------------------
  How to evaluate your model:
  ---------------------------
  Evaluation on test set should be conducted over full batch, i.e. 10k images,
  while it is alright to do it over minibatch for train set.

  ---------------------------------
  How often to evaluate your model:
  ---------------------------------
  - on training set every print_freq iterations
  - on test set every eval_freq iterations

  ------------------------
  Additional requirements:
  ------------------------
  Also you are supposed to take snapshots of your model state (i.e. graph,
  weights and etc.) every checkpoint_freq iterations. For this, you should
  study TensorFlow's tf.train.Saver class.
  """

  # Set the random seeds for reproducibility. DO NOT CHANGE.
  tf.set_random_seed(42)
  np.random.seed(42)
  tf.reset_default_graph()

  ########################
  # PUT YOUR CODE HERE  #
  ########################

  n_classes  = 10
  weight_decay = 1e-3
  weight_initializer =tf.random_normal_initializer(mean = 0 ,stddev = 1e-4, dtype = tf.float32)

  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

  x_train, y_train  = cifar10.train.images, cifar10.train.labels
  x_test,y_test = cifar10.test.images, cifar10.test.labels

  input_width = 32
  input_height = 32
  input_channels = 3

  #create placeholders
  x = tf.placeholder(tf.float32, shape = [FLAGS.batch_size,input_width,input_height,input_channels],name= 'train_data')
  y = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, n_classes], name="y-labels")
  test_x = tf.placeholder(tf.float32, shape = x_test.shape, name ='x-test')
  test_y = tf.placeholder(tf.int32, shape= y_test.shape,name = 'y-test')
  #create the CNN
  convnet = convnet_tf.ConvNet(n_classes,weight_initializer = weight_initializer,weight_decay = weight_decay)
  #Obtain logits
  logits = convnet.inference(x,reuse = False)
  #Obtain loss
  loss = convnet.loss(logits,y)
  #Select optimizer properly
  if (FLAGS.optimizer == 'ADAM'):
    optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
  elif (FLAGS.optimizer == 'SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = FLAGS.learning_rate)
  elif (FLAGS.optimizer == 'ADAGRAD'):
    optimizer = tf.train.AdagradOptimizer(learning_rate = FLAGS.learning_rate)
  elif (FLAGS.optimizer == 'ADADELTA'):
    optimizer = tf.train.Adadelta(learning_rate = FLAGS.learning_rate)
  else:
    optimizer = tf.train.RMSPropOptimizer(learning_rate = FLAGS.learning_rate)
  #training
  train_step = convnet.train_step(loss,optimizer)

  #train accuracy
  with tf.name_scope('train_accuracy'):
    train_acc = convnet.accuracy(logits,y)
  tf.summary.scalar('train_accuracy',train_acc)
  #test accuracy
  predictions = convnet.inference(test_x, reuse = True)
  with tf.name_scope('test_accuracy'):
    acc = convnet.accuracy(predictions,test_y)
  tf.summary.scalar('test_accuracy',acc)


  # Merge all the summaries and write them out to logdir
  merged = tf.summary.merge_all()

  #Create a saver
  saver = tf.train.Saver()

  #write in log directory
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',graph=tf.get_default_graph())
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  #initialize
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  for i in range(FLAGS.max_steps+1):
    batch_x,batch_y = next_batch(FLAGS.batch_size,x_train,y_train)
    if (i % FLAGS.eval_freq ==0):
        summary,l,accur = sess.run([merged,loss,acc], feed_dict = {x: batch_x, y:batch_y, test_x: x_test, test_y: y_test})
        print('Loss at step {}: {}'.format(i,l))
        print('Test accuracy at step {}: {}'.format(i,accur))
        test_writer.add_summary(summary,i)
    else:
        sess.run([train_step], feed_dict={x: batch_x, y:batch_y})
    if(i % FLAGS.print_freq == 0):
        train_ac = sess.run(train_acc,feed_dict = {x: batch_x, y:batch_y})
        print('Train_accuracy at step {}: {}'.format(i,train_ac))
    if (i % FLAGS.checkpoint_freq == 0):
        saver.save(sess, FLAGS.checkpoint_dir, global_step=i)

  train_writer.close()
  test_writer.close()
  sess.close()


  ########################
  # END OF YOUR CODE    #
  ########################

def initialize_folders():
  """
  Initializes all folders in FLAGS variable.
  """

  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)

  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  print_flags()

  initialize_folders()

  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
  parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
  parser.add_argument('--print_freq', type=int, default=PRINT_FREQ_DEFAULT,
                        help='Frequency of evaluation on the train set')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')
  parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
  parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')
  parser.add_argument('--optimizer', type=str, default=OPTIMIZER_DEFAULT,
                                            help='Optimizer')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
