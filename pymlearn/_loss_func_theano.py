# -*- coding: utf-8 -*-
import theano
from theano import tensor as T

__author__ = 'mkopersk'


def _compile_hinge_loss_func():
    W = T.matrix('W')
    X = T.matrix('X')
    Y = T.matrix('Y')
    scores_un = T.dot(W, X.T)
    scores_y = T.sum(scores_un * Y, axis=0)
    hinge = (T.sum(T.maximum(0, scores_un - scores_y + 1)) - Y.shape[1]) / X.shape[0]
    hinge_derivative = T.grad(hinge, W)
    return theano.function([W, X, Y], hinge), theano.function([W, X, Y], hinge_derivative)


def _compile_softmax_loss_func():
    W = T.matrix('W')
    X = T.matrix('X')
    Y = T.matrix('Y')
    scores = T.dot(W, X.T)
    scores_y = T.sum(scores * Y, axis=0)
    negative_log_likelihood = T.mean(-scores_y + T.log(T.sum(T.exp(scores), axis=0)))
    negative_log_likelihood_derviative = T.grad(negative_log_likelihood, W)
    return theano.function([W, X, Y], negative_log_likelihood), theano.function([W, X, Y],
                                                                                negative_log_likelihood_derviative)


def _compile_l1_penalty_func():
    W = T.matrix('W')
    l1_penalty_func = T.sum(T.abs_(W))
    l1_penalty_der_func = T.grad(l1_penalty_func, W)
    return theano.function([W], l1_penalty_func), theano.function([W], l1_penalty_der_func)


def _compile_l2_penalty_func():
    W = T.matrix('W')
    l2_penalty_func = T.sum(T.sqr(W))
    l2_penalty_der_func = T.grad(l2_penalty_func, W)
    return theano.function([W], l2_penalty_func), theano.function([W], l2_penalty_der_func)

hinge_loss, hinge_loss_derivatives = _compile_hinge_loss_func()
softmax_loss, softmax_loss_derivatives = _compile_softmax_loss_func()
l1_penalty, l1_penalty_der = _compile_l1_penalty_func()
l2_penalty, l2_penalty_der = _compile_l2_penalty_func()
