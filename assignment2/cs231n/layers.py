import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  -     here N=1000, d_1 = 3, d_2 = 32, d_3 = 32
  - w: A numpy array of weights, of shape (D, M)
  -     here D = 3x32x32, M is the # of nodes.
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0] # number of inputs
  dims = x.shape[1:] # all dimensions for each input (d_1,...,d_k)
  D = np.prod(dims) # product of all elements of dims
  # M = w.shape[1]
  xVect = x.reshape(N, D)  # each input x is now a vector.
  # f(x) = X.W + b
  out = xVect.dot(w) + b   # matrix multiplication + bias
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b) # cache original input, current weights and bias
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - and also b: biases, of shape (M,)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache # retrieve original input, last used weigths and bias
  dx, dw, db = None, None, None
  #############################################################################
  # Implement the affine backward pass.                                 #
  #############################################################################
  N = x.shape[0] # number of inputs
  dims = x.shape[1:]  # all dimensions for each input (d_1,...,d_k)
  D = np.prod(dims) # product of all elements of dims
  # dx = dout * w   -->   because df/dx = w, and we multiply it by the gradient we received from upstream
  dx = dout.dot(w.T).reshape(x.shape) # (N,M) * (D,M)^T = (N,D) --> reshape to (N, d1,..,dk)
  # dw = dout * x   -->   because df/dw = x, and we multiply it by the gradient we received from upstream
  xVect = x.reshape(N, D) # transform original x into a vector, as we did on affine_forward function
  dw = xVect.T.dot(dout)  # (N, D)^T * (N,M) = (D,M)  

  # db = dout * 1   --> because df/db = 1, and we multiply it by the gradient we received from upstream
  db = np.sum(dout,0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # Implement the ReLU forward pass.                                    #
  #############################################################################
  z = np.zeros_like(x)  # create a matrix of zeros, same size to x
  out = np.maximum(x,z) # write 0 instead of negative numbers
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = np.zeros_like(dout)
  
  # find positive indices of cached input
  posIdx = np.where(x>0)
  # create a mask
  dx[posIdx] = 1 
  # "gate" the upstream gradient according to mask
  dx = dx*dout 
  # from course notes: "max gate distributes the gradient (unchanged) to exactly 
  # one of its inputs (the input that had the highest value during the forward pass)."
  # here we do this for every sample with positive (cached) input value.
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    minibatch_mean = x.mean(axis = 0)
    minibatch_var = x.std(axis = 0)**2 # variance = std^2

    running_mean = momentum * running_mean + (1 - momentum) * minibatch_mean
    running_var = momentum * running_var + (1 - momentum) * minibatch_var

    # WE COULD DO IT LIKE THIS, 
    # x_normalized = (x - minibatch_mean) / np.sqrt(minibatch_var - eps)
    # out = gamma*x_normalized + beta

    # BUT LET'S DO IT STEP BY STEP TO EASE THE BACKPROP
    step1 = minibatch_mean # mean of x
    step2 = x - step1 # shift x by mean
    step3 = step2**2 
    step4 = np.sum(step3, axis=0) / float(N) # variance
    step5 = np.sqrt(step4 + eps)
    step6 = 1.0 / step5
    step7 = step2 * step6 # normalized x 
    step8 = gamma * step7
    out = step8 + beta # gamma*x_norm + beta, BN_gamma,beta(X) in the Szegedy paper        


    # pack vars into cache
    cache = (step1, step2, step3, step4, step5, step6, step7, step8, gamma, beta, x, bn_param)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_normalized = (x - running_mean) / np.sqrt(running_var - eps)
    out = gamma*x_normalized + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  N, D = dout.shape

  # unpack intermediates vars from cache
  step1, step2, step3, step4, step5, step6, step7, step8, gamma, beta, x, bn_param = cache
  # print 'step1 ', step1
  # print 'step2 ', step2
  # print 'step3 ', step3
  # print 'step4 ', step4
  # print 'step5 ', step5
  # print 'step6 ', step6
  # print 'step7 ', step7
  # print 'step8 ', step8

  eps = bn_param.get('eps', 1e-5)
  
  # following the chain rule, going 1 step back at a time:
  dstep8 = dout 
  dbeta = np.sum(dout, axis=0)  
  
  dgamma = np.sum(dstep8 * step7, axis=0) 
  dstep7 = gamma*dstep8 

  dstep6 = np.sum(step2 * dstep7, axis=0) 
  dstep2_part1 = step6*dstep7 
  dstep5 = -1.0 / (step5**2) * dstep6 # derivative of inverse times the upstream 
  dstep4 = 0.5 * (step4+eps)**(-0.5) * dstep5 # derivative of sqrt times the upstream
  dstep3 = 1.0 / float(N) * np.ones(step3.shape)*dstep4 # derivative of mean times the upstream
  dstep2_part2 = 2.0 * step2 * dstep3 # derivative of ()^2 times the upstream
  dstep2 = dstep2_part1 + dstep2_part2 
  dstep1 = -1.0 * np.sum(dstep2, axis=0) # -1 times upstream 
  dx_part1 = dstep2 # from the first branch (addition)
  dx_part2 = 1.0 / float(N) * np.ones(dstep2.shape) * dstep1 #  derivative of mean times upstream
  dx = dx_part1 + dx_part2 # sum of the branches


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta



def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  N, D = dout.shape

  # unpack cache
  step1, step2, step3, step4, step5, step6, step7, step8, gamma, beta, x, bn_param = cache

  eps = bn_param.get('eps', 1e-5)
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout * step7, axis=0) 
  dx = (1. / N) * gamma * (step4 + eps)**(-1. / 2.) * (N * dout - np.sum(dout, axis=0) - (x - step1) * (step4 + eps)**(-1.0) * np.sum(dout * (x - step1), axis=0))


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################

    mask = (np.random.rand(*x.shape) < p) / p # create the mask and scale it
    out = x*mask # apply the mask

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################

    out = x

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    
    # upstream gradient should flow according to mask
    dx = dout * mask

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # Implement the convolutional forward pass.                                 #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  N, C, H, W = x.shape # 2, 3, 4, 4
  F, C, HH, WW = w.shape # 3, 3, 4, 4
  stride = conv_param.get('stride', 1) # 2
  pad = conv_param.get('pad') # 1

  # pad the input array
  x_padded = np.lib.pad(x, ((0,),(0,),(pad,),(pad,)), 'constant') # default constant padding value is 0
  # print 'x_padded: ', x_padded.shape # (2, 3, 6, 6)
 
  # calculate output size
  Hout = 1 + (H + 2*pad - HH) / stride  # 2
  Wout = 1 + (W + 2*pad - WW) / stride  # 2

  # check if it's an integer
  try:
    int(Hout)
    int(Wout)   
  except ValueError:
    if not Hout.is_integer() or not Wout.is_integer():
      print 'Output sizes should be integers: ', Hout, Wout
      raise ValueError
    
  

  # initialize out array
  out = np.zeros((N, F, Hout, Wout))

  for i in xrange(0, Wout): 
    for j in xrange(0, Hout): # for each i,j coordinate of the output:                

        for n in xrange(0, N): # for each sample
          
          # cut out the x_padded for multiplication          
          input_data = x_padded[n, :, i*stride:i*stride+WW, j*stride:j*stride+HH] # volume of (3, WW, HH)
          # print 'cutout: ', n, ': (', i*stride, j*stride, ')------>(', i*stride+WW, j*stride+HH, ') x', C 

          for f in xrange(0, F):  # for each filter:                  
            # filter (kernel)
            kernel = w[f,] # volume of (3,WW, HH)             
            out[n, f, i, j] = np.sum(np.multiply(input_data, kernel)) + b[f]            


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # Implement the convolutional backward pass.                          #
  #############################################################################
  # unpack cache and fetch dimensions
  x, w, b, conv_param = cache
  stride = conv_param.get('stride', 1)
  pad = conv_param.get('pad')  
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  _, _, Hout, Wout = dout.shape

  # fix the size of the outputs
  dx = np.zeros(x.shape)
  dw = np.zeros(w.shape)
  db = np.zeros(b.shape)

  # pad the input and its gradient (will unpad it later)
  x_padded = np.lib.pad(x, ((0,),(0,),(pad,),(pad,)), 'constant') 
  dx_padded = np.lib.pad(dx, ((0,),(0,),(pad,),(pad,)), 'constant')

  # do the convolution again
  for i in xrange(0, Wout):
    for j in xrange(0, Hout): # for each coordinate of dout
      
      for n in xrange(0, N): # for each sample
        # cut out the x_padded for multiplication          
        input_data = x_padded[n, :, i*stride:i*stride+WW, j*stride:j*stride+HH] # volume of (3, WW, HH)
    
        for f in xrange(0, F): # for each filter       
          # compute the gradients:
          db[f] += dout[n, f, i, j] # it's accumulation of 1*upstream, since the biases are shared
          dw[f] += input_data * dout[n,f,i,j] # it's x*upstream (x being the relative part of the image, that we convolved earlier)
          dx_padded[n, :, i*stride:i*stride+WW, j*stride:j*stride+HH] += w[f,] * dout[n, f, i, j]


  # do the unpad 
  dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # Implement the max pooling forward pass                              #
  #############################################################################
  # fetch parameters
  pool_width = pool_param['pool_width']
  pool_height = pool_param['pool_height']
  stride = pool_param['stride']

  # get size of x
  (N, C, H, W) = x.shape

  # calculate output size
  Wout = (W-pool_width)/stride +1
  Hout = (H-pool_height)/stride +1
  out = np.zeros((N, C, Wout, Hout))

  # it's very similar to convolution
  for i in xrange(0, Wout):
    for j in xrange(0, Hout): # iterate over output coordinates
      for n in xrange(0, N): # for each sample
        for c in xrange(0, C): # for each channel
          out[n, c, i, j] = np.max(x[n, c, i*stride:i*stride+pool_width, j*stride:j*stride+pool_height])


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """

  # s = [aaa, bbb, cccc, ... , xxx]
  # L_i = sumForJisDifferentThanYi(max(0, s_j - s_yi + delta)), where yi is the score for correct label for ith sample 


  N = x.shape[0] # number of samples
  correct_class_scores = x[np.arange(N), y] # scores of correct classes
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0) # differences from all (including the correct classes)
  margins[np.arange(N), y] = 0 # set 0 for correct classes
  loss = np.sum(margins) / N

  # gradient of max(0,x)
  num_pos = np.sum(margins > 0, axis=1) # number of positive differences = number of "class scores that need adjustment"
  dx = np.zeros_like(x) # set all to 0
  dx[margins > 0] = 1 # mark for >zero inputs
  dx[np.arange(N), y] -= num_pos # subtract "nb of class scores that need adjustment" from correct classes ?
  dx /= N # normalize by the number of samples
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
