# -*- coding: utf-8 -*-
import numpy as np
from _loss_func_semi_vectorized import softmax_loss_derivatives, hinge_loss_derivatives, l1_penalty, l1_penalty_der, \
    l2_penalty, l2_penalty_der

__author__ = 'mkopersk'

print '[Warning] The derivatives are not implemented in fully vectorized way.'


def hinge_loss(W, X, Y):
    scores = np.dot(W, X.T)
    scores_y = np.sum(scores * Y, axis=0)
    hinge = (np.sum(np.maximum(0, scores - scores_y + 1)) - Y.shape[1]) / X.shape[0]
    return hinge


def softmax_loss(W, X, Y):
    scores = np.dot(W, X.T)
    scores_y = np.sum(scores * Y, axis=0)
    negative_log_likelihood = np.mean(-scores_y + np.log(np.sum(np.exp(scores), axis=0)))
    return negative_log_likelihood


_functions = {'softmax': (softmax_loss, softmax_loss_derivatives), 'hinge': (hinge_loss, hinge_loss_derivatives),
             'L1': (l1_penalty, l1_penalty_der), 'L2': (l2_penalty, l2_penalty_der)}


def get_loss_function(loss, penalty):
    loss_fun, loss_der_fun = _functions[loss]
    penalty_fun, penalty_der_fun = _functions[penalty]
    final_loss = lambda W, X, Y, reg: loss_fun(W.reshape(-1, X.shape[1]), X, Y) + reg * penalty_fun(W.reshape(-1, X.shape[1]))
    final_loss_der = lambda W, X, Y, reg: loss_der_fun(W.reshape(-1, X.shape[1]), X, Y) + reg * penalty_der_fun(W.reshape(-1, X.shape[1]))
    return final_loss, final_loss_der