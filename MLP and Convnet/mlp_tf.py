"""
This module implements a multi-layer perceptron in TensorFlow.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import l1_regularizer, l2_regularizer
import numpy as np

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in Tensorflow.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference, training and it
  can also be used for evaluating prediction performance.
  """

  def __init__(self, n_hidden, n_classes, is_training,weight_decay,
               activation_fn = tf.nn.relu, dropout_rate = 0.,
               weight_initializer = xavier_initializer(),
               weight_regularizer = l2_regularizer(0.001), inp_dim=3072):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter.

    Args:
      n_hidden: list of ints, specifies the number of units
                     in each hidden layer. If the list is empty, the MLP
                     will not have any hidden units, and the model
                     will simply perform a multinomial logistic regression.
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the MLP.
      is_training: Bool Tensor, it indicates whether the model is in training
                        mode or not. This will be relevant for methods that perform
                        differently during training and testing (such as dropout).
                        Have look at how to use conditionals in TensorFlow with
                        tf.cond.
      activation_fn: callable, takes a Tensor and returns a transformed tensor.
                          Activation function specifies which type of non-linearity
                          to use in every hidden layer.
      dropout_rate: float in range [0,1], presents the fraction of hidden units
                         that are randomly dropped for regularization.
      weight_initializer: callable, a weight initializer that generates tensors
                               of a chosen distribution.
      weight_regularizer: callable, returns a scalar regularization loss given
                               a weight variable. The returned loss will be added to
                               the total loss for training purposes.
    """
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.is_training = is_training
    self.activation_fn = activation_fn
    self.dropout_rate = dropout_rate
    self.weight_initializer = weight_initializer
    self.weight_regularizer = weight_regularizer

    self.weight_decay = weight_decay
    self._inp_dim = inp_dim
    self._structure = [inp_dim] + n_hidden +[n_classes]
    self._hidden_number = len(n_hidden)

    #initialize weights and biases
    
    self._weights ={}

    for i in range(0,self._hidden_number + 1):
        with tf.variable_scope("weights_{}".format(i)) as scope:
            new_layer = tf.get_variable(name = 'weights_{}'.format(i),shape = (self._structure[i],self._structure[i+1]), initializer = self.weight_initializer, regularizer = self.weight_regularizer)
            self._weights[i] = new_layer
            self.variable_summaries(self._weights[i])
            scope.reuse_variables()
    self._biases = [tf.zeros(y, tf.float32) for y in self._structure[1:]]
    
    self._Z = [None for i in range(self._hidden_number + 1)]
    self._S = [None for i in range(self._hidden_number + 1)]



#Variable summaries for tensorboard
  def variable_summaries(self,var):
     """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
     with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

  def inference(self, x):
    """
    Performs inference given an input tensor. This is the central portion
    of the network. Here an input tensor is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    In order to keep things uncluttered we recommend you (though it's not required)
    to implement a separate function that is used to define a fully connected
    layer of the MLP.

    In order to make your code more structured you can use variable scopes and name
    scopes. You can define a name scope for the whole model, for each hidden
    layer and for output. Variable scopes are an essential component in TensorFlow
    design for parameter sharing.

    You can use tf.summary.histogram to save summaries of the fully connected layer weights,
    biases, pre-activations, post-activations, and dropped-out activations
    for each layer. It is very useful for introspection of the network using TensorBoard.

    Args:
      x: 2D float Tensor of size [batch_size, input_dimensions]

    Returns:
      logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    self._input = x
    #Forward propagation
    for i in range(self._hidden_number):
        self._input, S = self.layer(self._input,i,'layer_{}'.format(i))
        self._Z[i] = self._input
        self._S[i] = S
    
    logits, S = self.layer(self._input,self._hidden_number,'layer_Out')
    self._S[-1] = S
    self._Z[-1] = logits
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return logits
  
  def layer(self,input,ind,name):
      #This function creates a layer
      
      with tf.name_scope(name):
          with tf.name_scope('Wx_plus_b'):
            S = tf.add(tf.matmul(input,self._weights[ind]), self._biases[ind])
            tf.summary.histogram('pre_activations',S)
          Z = self.activation_fn(S)
          tf.summary.histogram('activations',Z)
          return Z, S
  
  
  
  def loss(self, logits, labels):
    """
    Computes the multiclass cross-entropy loss from the logits predictions and
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
    #######################
    cross_loss = tf.nn.softmax_cross_entropy_with_logits( _sentinel=None, labels=labels, logits=logits, dim=-1, name=None)
    avg_loss = tf.reduce_mean(cross_loss, 0)
    loss = avg_loss + self.weight_decay * tf.nn.l2_loss(self._weights[self._hidden_number])
    ########################
    # END OF YOUR CODE    #
    #######################

    return loss

  def train_step(self,loss, optimizer):
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
    train_step = optimizer.minimize(loss)
    ########################
    # END OF YOUR CODE    #
    #######################

    return train_step

  def accuracy(self, logits, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    As in self.loss above, you can use tf.summary.scalar to save
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
    #######################

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy
