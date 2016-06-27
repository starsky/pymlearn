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
