from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    batch_size = X.shape[0]

    logits = X.dot(W)

    constant = np.max(logits)
    logits = np.exp(logits - constant)
    sum_logits = np.sum(logits, axis=1)

    delta = 1e-15
    for i in range(batch_size):
        logits[i] = logits[i] / sum_logits[i]
        # Note the minus sign.
        loss -= np.log(logits[i][y[i]] + delta)

    # Note the average over batch dimension.
    loss /= batch_size
   
    # gradient of softmax = y_pred - y_label
    d_logits = np.copy(logits)
    for i in range(batch_size):
        y_one_hot = np.zeros(num_classes)
        y_one_hot[y[i]] = 1
        d_logits[i] = (logits[i] - y_one_hot) / batch_size

    dW = X.T.dot(d_logits)

    # Note np.sum
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    batch_size = X.shape[0]

    logits = X.dot(W)
    c = np.max(logits)
    logits = np.exp(logits - c)
    sum_logits = np.sum(logits, axis=1, keepdims=True)

    logits = logits / sum_logits
    
    y_one_hot = np.zeros_like(logits)
    y_one_hot[np.arange(batch_size), y] = 1

    delta = 1e-15
    loss = np.sum(-np.log(logits + delta) * y_one_hot) / batch_size

    d_logits = (logits - y_one_hot) / batch_size
    dW = X.T.dot(d_logits)

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
