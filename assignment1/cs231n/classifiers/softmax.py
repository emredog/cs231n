import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  

  lossTraining = 0.0


  for i in xrange(num_train): # for each sample
    scores[i,:] += -np.max(scores[i,:]) # trick to keep numerical stability  

    # calculate e^f / Sum_c(e^f_c)    , where c is all classes
    normExpScores = np.exp(scores[i,:]) / np.sum(np.exp(scores[i,:]))

    # loss for i^th sample is -log(e^f_yi / Sum_c(e^f_c))
    # so we just need the correct index of normExpScores 
    lossTraining += -np.log(normExpScores[y[i]])


    # analytic gradient is just the multiplication of i'th input and e^f / Sum_c(e^f_c)
    # so it boils down to (e^f / Sum_c(e^f_c)) * X[i, :]
    # where shapes are (10 x 1) * (1x3073) = (10 x 3073)
    # and this signifies the contribution of i'th sample to parameter update
    dscore = np.reshape(normExpScores, (num_classes, 1)) * X[i, :]

    # we also have to add the d(-f_yi)/dW to the i'th row (equivalent form of loss function)
    # which is simply X[i,:]
    dscore[y[i],:] -=  X[i,:]

    dW += dscore.T # W.shape = (3073x10), so transpose dscore to fit it

    
  lossTraining /= scores.shape[0]
  lossRegularization = 0.5*reg*np.sum(W*W)

  dW = dW /num_train + reg * W

  loss = lossTraining + lossRegularization
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  scores += -np.max(scores) # trick to keep numerical stability  

  lossTraining = 0.0

  # calculate e^f / Sum_c(e^f_c)    , where c is all classes
  sumOfExps = np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
  normExpScores = np.exp(scores) / sumOfExps

  # loss for i^th sample is -log(e^f_yi / Sum_c(e^f_c))
  # so we just need the correct index of normExpScores 
  lossTraining = np.sum(-np.log(normExpScores[range(num_train), y]))

  # start with e^f / Sum_c(e^f_c)
  dscore = normExpScores
  # mark correct classes with ds-1 instead of ds
  dscore[range(num_train), y] -= 1
  # so that when we multiply, we'll also subtract X from those
  dW = dscore.T.dot(X)
    
  lossTraining /= scores.shape[0]
  lossRegularization = 0.5*reg*np.sum(W*W)

  dW = dW.T /num_train + reg * W

  loss = lossTraining + lossRegularization
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

