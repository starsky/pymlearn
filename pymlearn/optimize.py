# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
__author__ = 'mkopersk'


def gradient_descend(fun, x0, args=None, jac=None, learning_rate=1e-3, tol=1e-3, max_iter=1000, verbose=True):
    old_loss = np.inf
    if jac is None:
        jac = lambda params: compute_partial_derivatives_numerical(fun, params, args=args)
    x0 = x0.astype(np.float64)
    for i in range(max_iter):
        loss = fun(x0, *args)
        if verbose:
            print 'Iteration %d, loss value: %f' % (i, loss)
        if np.abs(old_loss - loss) < tol:
            break
        old_loss = loss
        x0 -= jac(x0, *args) * learning_rate
    return x0


def compute_partial_derivatives_numerical(func, parameters, args=None, step=1e-5):
    current_function_value = func(parameters, *args)
    partial_step = np.eye(parameters.ravel().shape[0]) * step
    new_parameters = parameters.ravel()[:, np.newaxis] + partial_step
    derivatives = map(lambda new_partial_param:
                      (func(new_partial_param.reshape(parameters.shape), *args)
                       - current_function_value) / step, new_parameters.T)
    return np.vstack(derivatives).sum(axis=1).reshape(parameters.shape)


def solve(solver, loss_func, params, args=None, jac=None, tol=1e-3, max_iter=1000):
    if solver == 'BFGS':
        r = scipy.optimize.minimize(loss_func, params, args=args, method='BFGS', jac=jac,
                                    options={'maxiter': max_iter}, tol=tol)
        return r['x']
    elif solver == 'GD':
        r = gradient_descend(loss_func, params, args=args, jac=jac, max_iter=max_iter, tol=tol)
        return r
    else:
        raise ValueError('Wrong solver name')
