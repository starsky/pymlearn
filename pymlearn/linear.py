# -*- coding: utf-8 -*-
import numpy as np
import optimize
import sklearn.base
import sklearn.metrics
from _loss_func_theano import *

# from IPython.core.debugger import Tracer
# import theano
# from sklearn import datasets
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import hinge_loss as hinge_loss_test
# from scipy.optimize import minimize


def classify(W, x):
    # add ones to make the bias trick
    x = np.hstack([x, np.ones(x.shape[0]).reshape((-1, 1))])
    scores = np.dot(W, x.T)
    labels = np.argmax(scores, axis=0)
    return labels


def _get_loss_func(loss, penalty, reg):
    predefined_loss_funcs = {'hinge': (hinge_loss, hinge_loss_derivatives),
                             'softmax': (softmax_loss, softmax_loss_derivatives)}
    predefined_penalty_funcs = {'L1': (l1_penalty, l1_penalty_der), 'L2': (l2_penalty, l2_penalty_der)}

    f, f_der = predefined_loss_funcs[loss]
    p, p_der = predefined_penalty_funcs[penalty]

    def loss_func(params, Xtr, Ytr):
        params = params.reshape((-1, Xtr.shape[1]))
        loss_val = softmax_loss(params, Xtr, Ytr) + reg * l2_penalty(params[:, :-1])
        return loss_val

    def loss_func_der(params, Xtr, Ytr):
        params = params.reshape((-1, Xtr.shape[1]))
        gradient = softmax_loss_derivatives(params, Xtr, Ytr) + \
                   reg * np.hstack([l2_penalty_der(params[:, :-1]), np.zeros((params.shape[0], 1))])
        return gradient.ravel()

    return loss_func, loss_func_der


def train_classifer(Xtr, Ytr, reg=1.0, loss='hinge', penalty='L2', max_iter=2000, tol=1e-3, solver='BFGS'):
    to_binary_label = sklearn.preprocessing.MultiLabelBinarizer()
    Y_bin = to_binary_label.fit_transform(Ytr[:, np.newaxis]).astype(theano.config.floatX).T
    Y_bin = Y_bin.astype(np.float32)

    X = np.hstack([Xtr, np.ones(Xtr.shape[0]).reshape((-1, 1))])
    loss_func, loss_func_der = _get_loss_func(loss, penalty, reg)
    params = np.random.random((len(np.unique(Ytr)), Xtr.shape[1] + 1))
    params_optimal = optimize.solve(solver, loss_func, params.ravel(), args=(X, Y_bin), jac=loss_func_der,
                                    tol=tol, max_iter=max_iter)
    params_optimal = params_optimal.reshape((len(np.unique(Ytr)), -1))
    return params_optimal


class LinearClassifer(sklearn.base.BaseEstimator):
    def __init__(self, reg=1.0, loss='hinge', penalty='L2', max_iter=2000, tol=1e-3, solver='BFGS'):
        self.reg = reg
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.params = None

    def fit(self, X, Y):
        self.params = train_classifer(X, Y, reg=self.reg, loss=self.loss, penalty=self.penalty,
                                      max_iter=self.max_iter, tol=self.tol, solver=self.solver)
        return self

    def predict(self, X):
        return classify(self.params, X)

    def score(self, X, Y):
        predicted = classify(self.params, X)
        return sklearn.metrics.accuracy_score(Y, predicted)
