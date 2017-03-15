# -*- coding: utf-8 -*-
import numpy as np
__author__ = 'mkopersk'


def softmax_loss(W, X, Y):
    Y_vec = np.argmax(Y, axis=0)
    loss = 0
    for x, y in zip(X, Y_vec):
        loss += __sofmax_loss_partial__(W, x, y)
    return loss / X.shape[0]


def __sofmax_loss_partial__(W, x, y):
    # Tracer()() #this one triggers the debugger
    scores = np.dot(W, x)
    scores -= scores.max()
    score_y = scores[y]
    return -score_y + np.log(np.exp(scores).sum())


def softmax_loss_derivatives(W, X, Y):
    Y_vec = np.argmax(Y, axis=0)
    D = np.zeros(W.shape)
    for x, y in zip(X, Y_vec):
        D += __softmax_loss_partial_derivatives__(W, x, y)
    return (D / X.shape[0]).ravel()


def __softmax_loss_partial_derivatives__(W, x, y):
    scores = np.dot(W, x)
    scores -= scores.max()
    mask = np.zeros(W.shape)
    mask[y] = 1
    return -x * mask + (np.exp(scores)[:, np.newaxis] * x) / (np.exp(scores).sum())


def hinge_loss(W, X, Y):
    Y_vec = np.argmax(Y, axis=0)
    loss = 0
    for x, y in zip(X, Y_vec):
        loss += __hinge_loss_partial__(W, x, y)
    return loss / X.shape[0]


def hinge_loss_derivatives(W, X, Y):
    Y_vec = np.argmax(Y, axis=0)
    D = np.zeros(W.shape)
    for x, y in zip(X, Y_vec):
        D += __hinge_loss_partial_derivatives__(W, x, y)
    return (D / X.shape[0]).ravel()


def __hinge_loss_partial_derivatives__(W, x, y):
    scores = np.dot(W, x)
    loss = scores - scores[y] + 1
    D = np.empty(W.shape)
    D[loss < 0] = 0
    D[loss == 0] = 0
    D[loss > 0] = x
    D[y] = np.zeros(x.shape)
    D[y] = -D.sum(axis=0)
    return D


def __hinge_loss_partial__(W, x, y):
    scores = np.dot(W, x)
    loss = scores - scores[y] + 1
    loss[y] = 0
    loss = np.maximum(0, loss)
    return loss.sum()


def l2_penalty(x):
    # x last column is a bias which we do not want do penalise
    return np.sum(x[:, :-1] ** 2)


def l2_penalty_der(x):
    # x last column is a bias which we do not want do penalise
    x_wo_bias = np.copy(x)
    x_wo_bias[:, -1] = 0
    return (2 * x_wo_bias).ravel()


def l1_penalty(x):
    # x last column is a bias which we do not want do penalise
    return np.sum(np.abs(x[:, :-1]))


def l1_penalty_der(x):
    D = np.zeros(x.shape)
    D[x < 0] = -1
    D[x > 0] = 1
    # x last column is a bias which we do not want do penalise
    D[:, -1] = 0
    return D.ravel()

_functions = {'softmax': (softmax_loss, softmax_loss_derivatives), 'hinge': (hinge_loss, hinge_loss_derivatives),
             'L1': (l1_penalty, l1_penalty_der), 'L2': (l2_penalty, l2_penalty_der)}


def get_loss_function(loss, penalty):
    loss_fun, loss_der_fun = _functions[loss]
    penalty_fun, penalty_der_fun = _functions[penalty]
    final_loss = lambda W, X, Y, reg: loss_fun(W.reshape(-1, X.shape[1]), X, Y) + reg * penalty_fun(W.reshape(-1, X.shape[1]))
    final_loss_der = lambda W, X, Y, reg: loss_der_fun(W.reshape(-1, X.shape[1]), X, Y) + reg * penalty_der_fun(W.reshape(-1, X.shape[1]))
    return final_loss, final_loss_der