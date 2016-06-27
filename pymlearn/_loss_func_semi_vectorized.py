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
    return D / X.shape[0]


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
    return D / X.shape[0]


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
    return np.sum(x ** 2)


def l2_penalty_der(x):
    return 2 * x


def l1_penalty(x):
    return np.sum(np.abs(x))


def l1_penalty_der(x):
    D = np.zeros(x.shape)
    D[x < 0] = -1
    D[x > 0] = 1
    return D
