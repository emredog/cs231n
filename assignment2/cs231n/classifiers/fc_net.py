import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
    self.params['b1'] = np.zeros((hidden_dim,))
    self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b2'] = np.zeros((num_classes,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # Implement the forward pass for the two-layer net, computing the          #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])   # 1st affine + ReLU
    out2, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])     # 2nd affine
    scores = np.maximum(0, out2)                                                  # softmax
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dout2 = softmax_loss(scores, y) # softmax loss

    dout1, dw2, db2 = affine_backward(dout2, cache2) # gradients & backward pass for 2nd affine    
    grads['W2'] = dw2 + self.reg * self.params['W2'] 
    grads['b2'] = db2

    dX, dw1, db1 = affine_relu_backward(dout1, cache1) # gradients & backward pass for 1st affine + ReLU
    grads['W1'] = dw1 + self.reg * self.params['W1'] 
    grads['b1'] = db1

    # loss + regularization
    loss += 0.5*self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.num_classes = num_classes

    ############################################################################
    # Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    # print 'Initializing with reg = ', reg, ', a network with ', self.num_layers, ' layers'
    for l in xrange(0,self.num_layers-1):
      if l == 0:
        self.params['W' + str(l+1)] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[l]))
      else:
        self.params['W' + str(l+1)] = np.random.normal(0, weight_scale, (hidden_dims[l-1], hidden_dims[l]))

      self.params['b' + str(l+1)] = np.zeros((hidden_dims[l],))
      
      if self.use_dropout:
        pass # TODO ?
      
      if self.use_batchnorm:
        self.params['gamma' + str(l+1)] = np.ones((1,)) #scale param gamma for l'th layer
        self.params['beta' + str(l+1)] = np.zeros((1,))  #shift param beta for l'th layer

    self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, (hidden_dims[self.num_layers-2], self.num_classes))
    self.params['b' + str(self.num_layers)] = np.zeros((self.num_classes,))


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    # print 'Computing loss...'
    ############################################################################
    # Implement the forward pass for the fully-connected net, computing        #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    caches = {}

    if self.use_dropout: # FIXME: maybe I can keep these caches elsewhere?
      dropout_caches = {}
    
    prevOut = []

    # reminder: {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    for l in xrange(1, self.num_layers):
      # print 'fwd: ', l            
      if (l == 1):
        inputData = X
      else:
        inputData = prevOut
      
      # print 'InputData: ', inputData.shape  
      Wcur = self.params['W' + str(l)]
      # print 'W' + str(l) + ' : ', Wcur.shape
      bcur = self.params['b' + str(l)]
      # print 'b' + str(l+1) + ' : ', bcur.shape

      if self.use_batchnorm:
        # fetch batchnorm-specific params for this layer
        gamma = self.params['gamma' + str(l)]
        beta = self.params['beta' + str(l)]
        bn_param = self.bn_params[l-1]
        # execute affine-batchnorm-relu fwd
        out, cache = affine_batchnorm_relu_forward(inputData, Wcur, bcur, gamma, beta, bn_param)
      else:
        out, cache = affine_relu_forward(inputData, Wcur, bcur)

      if self.use_dropout: # apply dropout and keep the cache
        out, dropout_cache = dropout_forward(out, self.dropout_param)
        dropout_caches[l-1] = dropout_cache
        # print 'FW dropout_cache[l-1] = dropout_cache[', l-1, ']'
        # _, masksilbeni = dropout_cache
        # print '     size of mask: ', masksilbeni.shape
        # print '     size of x: ', out.shape

      caches[l] = cache; # FIXME: is the first one (l=0) empty??

      prevOut = out
      

    # Final Layer
    Wcur = self.params['W' + str(self.num_layers)]    
    bcur = self.params['b' + str(self.num_layers)]    

    out, cache = affine_forward(prevOut, Wcur, bcur)  # affine final      
    caches[self.num_layers] = cache   
    scores = out
    


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    # reminder: {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    data_loss, dscores =  softmax_loss(scores, y) # softmax loss
    
    # backprop for the final affine layer
    finalCache = caches[self.num_layers]
    dout, dw, db = affine_backward(dscores, finalCache)
    grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
    grads['b' + str(self.num_layers)] = db

    # loss + regularization
    reg_loss = 0.5 * self.reg * np.sum(self.params['W' + str(self.num_layers)]**2)

    for l in xrange(self.num_layers-1, 0, -1):
      # print 'Bward: ', l+1
      cache = caches[l]

      if self.use_dropout:
        # print 'BW dropout_cache[l-1] = dropout_cache[', l-1, ']'
        # _, masksilbeni = dropout_caches[l-1]
        # print '     size of mask: ', masksilbeni.shape
        # print '     size of dx: ', dout.shape
        dout = dropout_backward(dout, dropout_caches[l-1])

      if self.use_batchnorm:
        dx, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, cache)
        grads['gamma' + str(l)] = np.sum(dgamma, axis=0)
        grads['beta' + str(l)] = np.sum(dbeta, axis=0)
      else:   
        dx, dw, db = affine_relu_backward(dout, cache)

      #print 'size of dx in l=', l, ' is ', dx.shape      

      dout = dx
      grads['W' + str(l)] = dw + self.reg * self.params['W' + str(l)]
      grads['b' + str(l)] = db
      
      # loss + regularization
      reg_loss += 0.5*self.reg*(np.sum(self.params['W' + str(l)]**2))      

    
    loss = data_loss + reg_loss
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


# THESE TWO are convenience layers with batchnorm, similar to ones in layer_utils.py 
# (it's explicitely stated that these two should be written in this file)
def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a batchnorm and by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma: scale param for batchnorm
  - beta: shift param for batchnorm
  - bn_param: dictionary with parameters for batchnorm

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  # fwd affine  
  a, fc_cache = affine_forward(x, w, b)
  # fwd batch_norm
  a_bnorm, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  # fwd relu
  out, relu_cache = relu_forward(a_bnorm)
  cache = (fc_cache, bn_cache, relu_cache)

  return out, cache


def affine_batchnorm_relu_backward(dout, cache):
  """
  Backward pass for the affine-batchnorm-relu convenience layer
  """
  # unpack cache
  fc_cache, bn_cache, relu_cache = cache  
  # backward relu
  da = relu_backward(dout, relu_cache)
  #backward batchnorm
  dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  #backward affine
  dx, dw, db = affine_backward(dbn, fc_cache)
  return dx, dw, db, dgamma, dbeta
