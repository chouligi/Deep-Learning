"""
This module implements a convolutional neural network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class ConvNet(object):
  """
  This class implements a convolutional neural network in TensorFlow.
  It incorporates a certain graph model to be trained and to be used
  in inference.
  """

  def __init__(self, n_classes = 10, weight_decay = 1e-3,weight_initializer = tf.random_normal_initializer(mean = 0 ,stddev = 1e-4, dtype = tf.float32)):
    """
    Constructor for an ConvNet object. Default values should be used as hints for
    the usage of each parameter.
    Args:
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the ConvNet.
    """
    self.n_classes = n_classes
  
    self.weight_initializer = weight_initializer
    self.weight_decay = weight_decay

  def inference(self, x, reuse):
    """
    Performs inference given an input tensor. This is the central portion
    of the network where we describe the computation graph. Here an input
    tensor undergoes a series of convolution, pooling and nonlinear operations
    as defined in this method. For the details of the model, please
    see assignment file.

    Here we recommend you to consider using variable and name scopes in order
    to make your graph more intelligible for later references in TensorBoard
    and so on. You can define a name scope for the whole model or for each
    operator group (e.g. conv+pool+relu) individually to group them by name.
    Variable scopes are essential components in TensorFlow for parameter sharing.
    Although the model(s) which are within the scope of this class do not require
    parameter sharing it is a good practice to use variable scope to encapsulate
    model.

    Args:
      x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
      reuse: Bool, indicates when to reuse the variables. reuse=False when we compute logits for training.
                reuse = True when computing logits of test

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
              the logits outputs (before softmax transformation) of the
              network. These logits can then be used with loss and accuracy
              to evaluate the model.
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    with tf.variable_scope('conv1',reuse = reuse) as scope:
        #convolution and relu
        conv1 = tf.layers.conv2d(x, 64, [5,5],strides = (1,1) ,activation=tf.nn.relu,kernel_initializer=self.weight_initializer)
        #max pooling
        conv1 = tf.layers.max_pooling2d(conv1,[3,3],[2,2])

    with tf.variable_scope('conv2',reuse = reuse) as scope:
        #convolution and relu
        conv2 = tf.layers.conv2d(conv1, 64, [5,5],strides = (1,1) ,activation=tf.nn.relu,kernel_initializer=self.weight_initializer)
        #max pooling
        conv2 = tf.layers.max_pooling2d(conv2,[3,3],[2,2])

    # Flatten the data to a 1-D vector for the fully connected layer
    with tf.variable_scope('fc1',reuse = reuse) as scope:
        fc1 = tf.contrib.layers.flatten(conv2)
        #apply relu to fully connected layer 1
        fc1 = tf.layers.dense(fc1,384,activation = tf.nn.relu,kernel_initializer=self.weight_initializer)

    with tf.variable_scope('fc2',reuse = reuse) as scope:
        fc2 = tf.layers.dense(fc1,192,activation = tf.nn.relu,kernel_initializer=self.weight_initializer)

    with tf.variable_scope('fc3',reuse = reuse) as scope:
        with tf.name_scope('logits'):
            logits = tf.layers.dense(fc2,self.n_classes,activation = None)
        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(fc2,self.n_classes,activation = tf.nn.softmax)
    ########################
    # END OF YOUR CODE    #
    ########################
    return logits

  def loss(self, logits, labels):
    """
    Calculates the multiclass cross-entropy loss from the logits predictions and
    the ground truth labels. The function will also add the regularization
    loss from network weights to the total loss that is return.

    In order to implement this function you should have a look at
    tf.nn.softmax_cross_entropy_with_logits.
    
    You can use tf.summary.scalar to save scalar summaries of
    cross-entropy loss, regularization loss, and full loss (both summed)
    for use with TensorBoard. This will be useful for compiling your report.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.

    Returns:
      loss: scalar float Tensor, full loss = cross_entropy + reg_loss
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################

    #get weights of the final layer
    output_weights = tf.get_default_graph().get_tensor_by_name("fc3/dense/kernel:0")

    with tf.name_scope('cross_entropy'):
        cross_loss = tf.nn.softmax_cross_entropy_with_logits( _sentinel=None, labels=labels, logits=logits, dim=-1, name='cross_loss')
    with tf.name_scope('average_loss'):
        avg_loss = tf.reduce_mean(cross_loss, 0)
    with tf.name_scope('total_loss'):
        loss = tf.add(avg_loss , self.weight_decay * tf.nn.l2_loss(output_weights))
    tf.summary.scalar('loss',loss)


    ########################
    # END OF YOUR CODE    #
    ########################

    return loss

  def train_step(self, loss, optimizer):
    """
    Implements a training step using a parameters in flags.

    Args:
      loss: scalar float Tensor.
      flags: contains necessary parameters for optimization.
    Returns:
      train_step: TensorFlow operation to perform one training step
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    with tf.name_scope('train'):
        train_step = optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    #######################

    return train_step

  def accuracy(self, logits, labels):
    """
    Calculate the prediction accuracy, i.e. the average correct predictions
    of the network.
    As in self.loss above, you can use tf.scalar_summary to save
    scalar summaries of accuracy for later use with the TensorBoard.

    Args:
      logits: 2D float Tensor of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int Tensor of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.

    Returns:
      accuracy: scalar float Tensor, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)
    ########################
    # END OF YOUR CODE    #
    ########################

    return accuracy

