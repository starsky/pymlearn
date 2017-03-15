# -*- coding: utf-8 -*-
import numpy as np
import optimize
import sklearn.base
import sklearn.metrics
import loss_functions
import theano
import core
from lasagne.updates import sgd

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


def train_classifer(Xtr, Ytr, reg=1.0, loss='hinge', penalty='L2', max_iter=2000, tol=1e-3, solver='BFGS',
                    verbose=False):
    to_binary_label = sklearn.preprocessing.MultiLabelBinarizer()
    Y_bin = to_binary_label.fit_transform(Ytr[:, np.newaxis]).astype(theano.config.floatX).T
    Y_bin = Y_bin.astype(np.float32)

    X = np.hstack([Xtr, np.ones(Xtr.shape[0]).reshape((-1, 1))])
    params = np.random.random((len(np.unique(Ytr)), Xtr.shape[1] + 1))
    if core.loss_backend == 'theano' and core.solver_backend == 'theano':
        loss_func, loss_func_der, symbolic_params = loss_functions.get_theano_not_compiled_loss(loss, penalty, params.ravel())
        updates = sgd(loss_func, [symbolic_params[0]], learning_rate=1e-3)
        train_fun = theano.function(symbolic_params[1:], loss_func, updates=updates)
        loss_fn_compiled = theano.function(symbolic_params[1:], loss_func)
        current_err = 1e6
        for i in range(10000):
            train_fun(X, Y_bin, reg)
            err =  loss_fn_compiled(X, Y_bin, reg)
            if abs(current_err - err) < 1e-5:
                break
            current_err = err
            print '%d\t%f' % (i, current_err)
    else:
        loss_func, loss_func_der = loss_functions.get_loss_function(loss, penalty)
        train_fun = optimize.solve(solver, loss_func, jac=loss_func_der, tol=tol, max_iter=max_iter, verbose=verbose)
        params_optimal = train_fun(params.ravel(), args=(X, Y_bin, reg))['x']
    # params_optimal = params_optimal.reshape((len(np.unique(Ytr)), -1))
    return symbolic_params[0]


class LinearClassifer(sklearn.base.BaseEstimator):
    def __init__(self, reg=1.0, loss='hinge', penalty='L2', max_iter=2000, tol=1e-3, solver='BFGS', verbose=False):
        self.reg = reg
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.params = None
        self.verbose = verbose

    def fit(self, X, Y):
        self.params = train_classifer(X, Y, reg=self.reg, loss=self.loss, penalty=self.penalty,
                                      max_iter=self.max_iter, tol=self.tol, solver=self.solver, verbose=self.verbose)
        return self

    def predict(self, X):
        return classify(self.params, X)

    def score(self, X, Y):
        predicted = classify(self.params, X)
        return sklearn.metrics.accuracy_score(Y, predicted)
