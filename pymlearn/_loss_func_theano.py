# -*- coding: utf-8 -*-
import theano
from theano import tensor as T

__author__ = 'mkopersk'


def _prepare_input_tensors(W=None, X=None, Y=None):
    W = T.vector('W') if W is None else W
    X = T.matrix('X') if X is None else X
    Y = T.matrix('Y') if Y is None else Y
    return W, X, Y


def _compile_hinge_loss_func(W=None, X=None, Y=None, compile_=False):
    """
    Prepares loss function and its derivative.
    :param W: Params as a matrix of dimension <#labels>X<#features> (#features = features size + bias)
    :param X: Input features as matrix <#examples>X<#features>
    :param Y: Input labels as matrix <#examples>X<#labels> (one hot vector)
    :param compile_: If true returned functions will be compiled
    :return: (Loss function, Loss function derivative)
    """
    W, X, Y = _prepare_input_tensors(W, X, Y)
    scores_un = T.dot(W, X.T)
    scores_y = T.sum(scores_un * Y, axis=0)
    hinge = (T.sum(T.maximum(0, scores_un - scores_y + 1)) - Y.shape[1]) / X.shape[0]
    hinge_derivative = T.grad(hinge, W).ravel()
    if compile_:
        return theano.function([W, X, Y], hinge), theano.function([W, X, Y], hinge_derivative)
    return hinge, hinge_derivative


def _compile_softmax_loss_func(W=None, X=None, Y=None, compile_=False):
    """
    Prepares loss function and its derivative.
    :param W: Params as a matrix of dimension <#labels>X<#features> (#features = features size + bias)
    :param X: Input features as matrix <#examples>X<#features>
    :param Y: Input labels as matrix <#examples>X<#labels> (one hot vector)
    :param compile_: If true returned functions will be compiled
    :return: (Loss function, Loss function derivative)
    """
    W, X, Y = _prepare_input_tensors(W, X, Y)
    scores = T.dot(W, X.T)
    scores_y = T.sum(scores * Y, axis=0)
    negative_log_likelihood = T.mean(-scores_y + T.log(T.sum(T.exp(scores), axis=0)))
    negative_log_likelihood_derviative = T.grad(negative_log_likelihood, W).ravel()
    if compile_:
        return theano.function([W, X, Y], negative_log_likelihood), \
               theano.function([W, X, Y], negative_log_likelihood_derviative)
    return negative_log_likelihood, negative_log_likelihood_derviative


def _compile_l1_penalty_func(W=None, compile_=False):
    """
    Prepares penalty function and its derivative.
    :param W: Params as a matrix of dimension <#labels>X<#features> (#features = features size + bias)
    :param compile_:  If true returned functions will be compiled
    :return: (Penalty function, Penalty function derivative)
    """
    W, _, _ = _prepare_input_tensors(W)
    W_wo_bias = W[:, :-1]
    l1_penalty_func = T.sum(T.abs_(W_wo_bias))
    l1_penalty_der_func = T.concatenate([T.grad(l1_penalty_func, W_wo_bias), T.zeros((W.shape[0], 1))], axis=1).ravel()
    if compile_:
        return theano.function([W], l1_penalty_func), theano.function([W], l1_penalty_der_func)
    return l1_penalty_func, l1_penalty_der_func


def _compile_l2_penalty_func(W=None,compile_=False):
    """
    Prepares penalty function and its derivative.
    :param W: Params as a matrix of dimension <#labels>X<#features> (#features = features size + bias)
    :param compile_:  If true returned functions will be compiled
    :return: (Penalty function, Penalty function derivative)
    """
    W, _, _ = _prepare_input_tensors(W)
    W_wo_bias = W[:, :-1]
    l2_penalty_func = T.sum(T.sqr(W_wo_bias))
    l2_penalty_der_func = T.concatenate([T.grad(l2_penalty_func, W_wo_bias), T.zeros((W.shape[0], 1))], axis=1).ravel()
    if compile_:
        return theano.function([W], l2_penalty_func), theano.function([W], l2_penalty_der_func)
    return l2_penalty_func, l2_penalty_der_func

_functions = {'softmax': _compile_softmax_loss_func, 'hinge': _compile_hinge_loss_func,
             'L1': _compile_l1_penalty_func, 'L2': _compile_l2_penalty_func}


def get_loss_function(loss, penalty):
    """
    Returns loss function and loss function derivative composed of selected loss and penalty function.
    Both loss_function and loss_function_derivateve input parameters are W,X,Y,reg. Where
    W - model params vector. It is a vector obtained from matrix <#labels>X<#features>, <#features> = features size + bias
    X - Input features as matrix <#examples>X<#features>
    Y - Input labels as matrix <#examples>X<#labels> (one hot vector)
    reg - regularization param manges influence of penalty function as follows:
        final_loss_function = loss_function + reg * penalty_function
    :param loss: 'softmax' or 'hinge'
    :param penalty: 'L1' or 'L2'
    :return: loss_function, loss_function_derivative (both returned objects are python functions)
    """
    W, X, Y = _prepare_input_tensors()
    W_reshaped = T.reshape(W, (-1, X.shape[1]))
    reg = T.scalar('reg')
    loss_function, loss_function_der = _functions[loss](W_reshaped, X, Y)
    penalty_function, penalty_function_der = _functions[penalty](W_reshaped)
    final_loss_function = loss_function + reg * penalty_function
    final_loss_function_der = loss_function_der + reg * penalty_function_der
    return theano.function([W, X, Y, reg], final_loss_function), theano.function([W, X, Y, reg],
                                                                                 final_loss_function_der)
