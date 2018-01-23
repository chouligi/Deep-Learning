"""
This module implements a multi-layer perceptron in NumPy.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform inference, training and it
  can also be used for evaluating prediction performance.
  """

  def __init__(self, n_hidden, n_classes, weight_decay, weight_scale,dim = 3072):
    """
    Constructor for an MLP object. Default values should be used as hints for
    the usage of each parameter. Weights of the linear layers should be initialized
    using normal distribution with mean = 0 and std = weight_scale. Biases should be
    initialized with constant 0. All activation functions are ReLUs.

    Args:
      n_hidden: list of ints, specifies the number of units
                     in each hidden layer. If the list is empty, the MLP
                     will not have any hidden units, and the model
                     will simply perform a multinomial logistic regression.
      n_classes: int, number of classes of the classification problem.
                      This number is required in order to specify the
                      output dimensions of the MLP.
      weight_decay: L2 regularization parameter for the weights of linear layers.
      weight_scale: scale of normal distribution to initialize weights.

    """
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.weight_decay = weight_decay
    self.weight_scale = weight_scale
    self.hidden_number = len(self.n_hidden)
    self.weights = []
    self._dL_dw = None
    self._input = None
  
    self.structure = [dim] + self.n_hidden +[self.n_classes]
    self.weights = [np.random.normal(loc =0.0,scale=self.weight_scale, size=(x,y)) for x,y in zip(self.structure[:-1],self.structure[1:])]
    
    self.weights = np.array(self.weights)

    self.biases = [np.zeros(y) for y in self.structure[1:]]
    self.biases = np.array(self.biases)

        #list to store activations for every layer
    self.activations = [None for i in range(self.hidden_number + 1)]
        #list to store outputs for every layer
    self.outs = [None for i in range(self.hidden_number + 1)]


  def inference(self, x):
    """
    Performs inference given an input array. This is the central portion
    of the network. Here an input tensor is transformed through application
    of several hidden layer transformations (as defined in the constructor).
    We recommend you to iterate through the list self.n_hidden in order to
    perform the sequential transformations in the MLP. Do not forget to
    add a linear output layer (without non-linearity) as the last transformation.

    It can be useful to save some intermediate results for easier computation of
    gradients for backpropagation during training.
    Args:
      x: 2D float array of size [batch_size, input_dimensions]

    Returns:
      logits: 2D float array of size [batch_size, self.n_classes]. Returns
             the logits outputs (before softmax transformation) of the
             network. These logits can then be used with loss and accuracy
             to evaluate the model.
    """


    self._input = x
    activation = self._input
    #input dimensions
    inp_dim = self._input.shape[1]
    self.batch_size = self._input.shape[0]
    for i in range(self.hidden_number):
        out =  self.biases[i] + activation@self.weights[i]
        activation = self.relu(out)
        self.outs[i] = out
        self.activations[i] = activation
    logits = activation@self.weights[-1]
    self.outs[-1] = logits
    self.activations[-1] = logits
    return logits
  
  def softmax2(self,x):
    exp_x = np.exp(x.T-np.max(x,axis = 1))
    return (exp_x / np.sum(exp_x,axis=0)).T


  def loss(self, logits, labels):
        """
        Computes the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.

        It can be useful to compute gradients of the loss for an easier computation of
        gradients for backpropagation during training.

        Args:
        logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
        labels: 2D int array of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.
        Returns:
        loss: scalar float, full loss = cross_entropy + reg_loss
        """
        #the derivative of the loss
        y_out = self.softmax2(logits)

        
        #self._dL_dw = self.cross_grad(logits,labels)
        self._dL_dw = y_out - labels
        #add small pertubation in order not to have 0 in the logarithm
        per = 10e-200
        cross_entropy = -np.sum(np.multiply(labels, np.log(y_out+per)))/self.batch_size
        #loss = cross_entropy + regularization
        loss = cross_entropy  +  self.weight_decay* 0.5 * np.linalg.norm(self.weights[-1])**2

        return loss

  
  def d_relu(self,x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
  def relu(self,x):
    #implement relu
    r = np.maximum(0,x)
    return r
  def train_step(self, loss, flags):
    """
    Implements a training step using a parameters in flags.
    Use Stochastic Gradient Descent to update the parameters of the MLP.

    Args:
      loss: scalar float Tensor.
      flags: contains necessary parameters for optimization.
    Returns:

    """
    #initialize gradients
    dw = [None for el in range(self.hidden_number+1)]
    db = [None for el in range(self.hidden_number+1)]
    #initialize delta
    delta = [None for i in range(self.hidden_number +1)]
    #delta_out
    delta[-1] = self._dL_dw
    #deltas for the hidden layers
    for i in range(2,len(self.structure)):
        delta[-i] = (delta[-i+1]@self.weights[-i+1].T)*self.d_relu(self.outs[-i])
    #calculate the gradients of hidden layers
    for i in range(len(dw)-1,0,-1):
        dw[i] = self.activations[i-1].T@delta[i] + self.weight_decay * self.weights[i]
    dw[0] = self._input.T@delta[0] + self.weight_decay * self.weights[0]
    for i in range(len(db)-1,0,-1):
        db[i] = np.sum(delta[i], axis = 0)
    db[0] = np.sum(delta[0],axis = 0)

    #perform the updates
    self.weights -= flags/self.batch_size * np.array(dw)
    self.biases -= flags/self.batch_size* np.array(db)
    return

  def accuracy(self, logits, labels):
    """
            Computes the prediction accuracy, i.e. the average of correct predictions
            of the network.

        Args:
        logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
                   labels: 2D int array of size [batch_size, self.n_classes]
                 with one-hot encoding. Ground truth labels for
                 each sample in the batch.
                 Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #get indices of maximum values along the rows
    pred = np.argmax(logits,axis = 1)
    #get indices of the labels, since they are one hot
    truth = np.argmax(labels,axis =1)
    accuracy = np.average(pred == truth)
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


